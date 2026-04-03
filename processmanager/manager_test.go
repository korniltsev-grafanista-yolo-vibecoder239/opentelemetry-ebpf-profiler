package processmanager

import (
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"

	lru "github.com/elastic/go-freelru"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/ebpf-profiler/host"
	"go.opentelemetry.io/ebpf-profiler/interpreter"
	golang "go.opentelemetry.io/ebpf-profiler/interpreter/go"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
	"go.opentelemetry.io/ebpf-profiler/process"
	"go.opentelemetry.io/ebpf-profiler/remotememory"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/util"
)

// traceCapture is a mock TraceReporter that records reported traces.
type traceCapture struct {
	traces []*libpf.Trace
}

func (tc *traceCapture) ReportTraceEvent(trace *libpf.Trace, _ *samples.TraceEventMeta) error {
	tc.traces = append(tc.traces, trace)
	return nil
}

// TestFrameCacheCrossProcessPollution verifies that a native frame from libc
// is not incorrectly symbolized as a Go frame and then served from the cache
// to unrelated processes.
//
// The test sends two EbpfTraces through HandleTrace — one for a Go process,
// one for a plain C process ("cat") — both containing an identical native
// frame with libc's file ID at an address that collides with a Go pclntab
// entry. Neither trace should contain a GoFrame.
func TestFrameCacheCrossProcessPollution(t *testing.T) {
	require := require.New(t)

	exec, err := os.Executable()
	require.NoError(err)

	pc, _, _, ok := runtime.Caller(0)
	require.True(ok)
	goFuncName := runtime.FuncForPC(pc).Name()

	goPID := libpf.PID(1000)
	catPID := libpf.PID(2000)

	goHostFileID, err := host.FileIDFromBytes(
		[]byte{0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55})
	require.NoError(err)
	catHostFileID, err := host.FileIDFromBytes(
		[]byte{0xCA, 0x7C, 0xA7, 0xCA, 0x7C, 0xA7, 0xCA, 0x7C})
	require.NoError(err)
	libcHostFileID, err := host.FileIDFromBytes(
		[]byte{0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE})
	require.NoError(err)

	// Load the real Go interpreter from the test binary.
	realPID := libpf.PID(os.Getpid())
	pid := process.New(realPID, realPID)
	elfRef := pfelf.NewReference(exec, pid)
	loaderInfo := interpreter.NewLoaderInfo(goHostFileID, elfRef)
	rm := remotememory.NewProcessVirtualMemory(realPID)

	goData, err := golang.Loader(nil, loaderInfo)
	require.NoError(err)
	goInstance, err := goData.Attach(nil, realPID, 0x0, rm)
	require.NoError(err)

	goODID := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 1}

	frameCache, err := lru.New[frameCacheKey, libpf.Frames](1024, hashFrameCacheKey)
	require.NoError(err)
	frameCache.SetLifetime(frameCacheLifetime)

	// Build per-process mappings, sorted by (FileID, Vaddr).
	goMappings := []Mapping{
		{FrameMapping: libpf.NewFrameMapping(libpf.FrameMappingData{
			File: libpf.NewFrameMappingFile(libpf.FrameMappingFileData{
				FileID:   libpf.NewFileID(uint64(goHostFileID), 0),
				FileName: libpf.Intern("go-binary"),
			}),
			Start: 0,
			End:   0xFFFFFFF,
		})},
		{FrameMapping: libpf.NewFrameMapping(libpf.FrameMappingData{
			File: libpf.NewFrameMappingFile(libpf.FrameMappingFileData{
				FileID:   libpf.NewFileID(uint64(libcHostFileID), 0),
				FileName: libpf.Intern("libc.so.6"),
			}),
			Start: 0,
			End:   0xFFFFFFF,
		})},
	}
	slices.SortFunc(goMappings, compareMapping)

	catMappings := []Mapping{
		{FrameMapping: libpf.NewFrameMapping(libpf.FrameMappingData{
			File: libpf.NewFrameMappingFile(libpf.FrameMappingFileData{
				FileID:   libpf.NewFileID(uint64(catHostFileID), 0),
				FileName: libpf.Intern("cat"),
			}),
			Start: 0,
			End:   0xFFFFFFF,
		})},
		{FrameMapping: libpf.NewFrameMapping(libpf.FrameMappingData{
			File: libpf.NewFrameMappingFile(libpf.FrameMappingFileData{
				FileID:   libpf.NewFileID(uint64(libcHostFileID), 0),
				FileName: libpf.Intern("libc.so.6"),
			}),
			Start: 0,
			End:   0xFFFFFFF,
		})},
	}
	slices.SortFunc(catMappings, compareMapping)

	capture := &traceCapture{}
	pm := &ProcessManager{
		interpreters: map[libpf.PID]map[util.OnDiskFileIdentifier]interpreter.Instance{
			goPID: {goODID: goInstance},
			// catPID has NO interpreters — it's a plain C program.
		},
		pidToProcessInfo: map[libpf.PID]*processInfo{
			goPID:  {mappings: goMappings},
			catPID: {mappings: catMappings},
		},
		frameCache:    frameCache,
		traceReporter: capture,
	}

	// Build the eBPF frame data: a single native frame at a Go function's
	// file VA but carrying libc's file ID. This is what the BPF unwinder
	// produces when sampling a libc function whose file offset happens to
	// collide with a Go pclntab entry.
	frameData := libpf.NewEbpfFrame(libpf.NativeFrame, 0, 2, uint64(pc))
	frameData[1] = uint64(libcHostFileID)

	// Step 1: HandleTrace for the Go process.
	pm.HandleTrace(&libpf.EbpfTrace{
		PID:       goPID,
		TID:       goPID,
		NumFrames: 1,
		FrameData: frameData,
	})

	require.Len(capture.traces, 1, "expected one reported trace for the Go process")
	goTrace := capture.traces[0]
	require.NotEmpty(goTrace.Frames, "Go process trace should have frames")

	goFrame := goTrace.Frames[0].Value()
	t.Logf("Go process frame: type=%v func=%q", goFrame.Type, goFrame.FunctionName)

	// The frame belongs to libc, not the Go binary. It must remain a native
	// frame and must not carry a Go function name.
	assert.Equal(t, libpf.NativeFrame, goFrame.Type,
		"libc frame in Go process must stay NativeFrame, not GoFrame")
	assert.False(t, strings.HasPrefix(goFrame.FunctionName.String(), goFuncName),
		"libc frame must not get Go function name %q", goFuncName)

	// Step 2: HandleTrace for the cat process with the exact same frame data.
	pm.HandleTrace(&libpf.EbpfTrace{
		PID:       catPID,
		TID:       catPID,
		NumFrames: 1,
		FrameData: frameData,
	})

	require.Len(capture.traces, 2, "expected two reported traces total")
	catTrace := capture.traces[1]
	require.NotEmpty(catTrace.Frames, "cat process trace should have frames")

	catFrame := catTrace.Frames[0].Value()
	t.Logf("Cat process frame: type=%v func=%q", catFrame.Type, catFrame.FunctionName)

	// The cat process is a plain C program. Its libc frame must also be a
	// plain native frame without any Go symbolization.
	assert.Equal(t, libpf.NativeFrame, catFrame.Type,
		"libc frame in cat process must be NativeFrame, not GoFrame")
	assert.False(t, strings.HasPrefix(catFrame.FunctionName.String(), goFuncName),
		"cat process must not inherit Go function name %q from cache", goFuncName)
}
