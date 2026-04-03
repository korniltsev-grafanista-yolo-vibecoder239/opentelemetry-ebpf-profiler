package processmanager

import (
	"os"
	"runtime"
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

// TestGoInterpreterSymbolizesForeignFileID demonstrates the bug where the Go
// interpreter incorrectly symbolizes a native frame from a different file
// (e.g. libc) because it only checks the frame type (Native) but not the
// file ID. When the file-relative address in the foreign file coincidentally
// matches a Go function's pclntab entry, the frame gets wrong Go symbols.
//
// The frame cache then amplifies the problem: since native frames are cached
// without PID in the key, a subsequent process (without any Go interpreter)
// receives the incorrectly symbolized frame from cache.
func TestGoInterpreterSymbolizesForeignFileID(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	// Use the test binary itself as the Go binary. Its pclntab is real
	// and contains all runtime functions at their actual file VA addresses.
	exec, err := os.Executable()
	require.NoError(err)

	// Get the file VA of a known Go function (runtime.Caller) so we can
	// construct a frame at that address but with a *different* file ID,
	// simulating a libc frame whose file offset collides with a Go function.
	pc, _, _, ok := runtime.Caller(0)
	require.True(ok, "failed to get PC")

	// Set up the Go interpreter from the test binary.
	goPID := libpf.PID(os.Getpid())
	pid := process.New(goPID, goPID)
	elfRef := pfelf.NewReference(exec, pid)

	goFileID, err := host.FileIDFromBytes([]byte{0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55})
	require.NoError(err)

	loaderInfo := interpreter.NewLoaderInfo(goFileID, elfRef)
	rm := remotememory.NewProcessVirtualMemory(goPID)

	goData, err := golang.Loader(nil, loaderInfo)
	require.NoError(err)

	goInstance, err := goData.Attach(nil, goPID, 0x0, rm)
	require.NoError(err)

	// Construct a native frame at the same address as the Go function,
	// but with a completely different file ID (representing libc).
	libcFileID, err := host.FileIDFromBytes([]byte{0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE})
	require.NoError(err)

	ef := libpf.NewEbpfFrame(libpf.NativeFrame, 0, 2, uint64(pc))
	ef[1] = uint64(libcFileID) // Variable(0) = libc file ID, NOT the Go binary

	// Try to symbolize this "libc" frame through the Go interpreter.
	var frames libpf.Frames
	err = goInstance.Symbolize(ef, &frames, libpf.FrameMapping{})

	// BUG: The Go interpreter succeeds, producing a Go symbol for a frame
	// that belongs to a completely different file. It should have either
	// returned ErrMismatchInterpreterType or checked the file ID first.
	if err == nil {
		// The test demonstrates the bug exists: a libc frame got Go symbols.
		assert.NotEmpty(frames, "Go interpreter produced frames for a foreign file ID")

		if len(frames) > 0 {
			f := frames[0].Value()
			t.Logf("BUG: Go interpreter claimed a frame with fileID=%x (not the Go binary's %x)",
				libcFileID, goFileID)
			t.Logf("  Produced: type=%v func=%q file=%q",
				f.Type, f.FunctionName, f.SourceFile)
			assert.Equal(libpf.GoFrame, f.Type,
				"frame was rewritten to GoFrame despite belonging to a different file")
		}
	} else {
		// If this branch is reached, the bug is fixed: the Go interpreter
		// correctly rejected the frame because the file ID didn't match.
		t.Log("Go interpreter correctly rejected the foreign frame")
	}
}

// traceCapture is a mock TraceReporter that records the last reported trace.
type traceCapture struct {
	traces []*libpf.Trace
}

func (tc *traceCapture) ReportTraceEvent(trace *libpf.Trace, _ *samples.TraceEventMeta) error {
	tc.traces = append(tc.traces, trace)
	return nil
}

// TestFrameCacheCrossProcessPollution demonstrates that the frame cache
// serves incorrectly symbolized frames to unrelated processes because
// native frames use PID-agnostic cache keys.
//
// The test sends two EbpfTraces through HandleTrace — one for a Go process,
// one for a plain C process ("cat") — both containing an identical native
// frame with libc's file ID at an address that collides with a Go pclntab
// entry. The Go process's trace gets incorrectly symbolized by the Go
// interpreter, and that result is cached and served to the cat process.
func TestFrameCacheCrossProcessPollution(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	exec, err := os.Executable()
	require.NoError(err)

	pc, _, _, ok := runtime.Caller(0)
	require.True(ok)

	goPID := libpf.PID(1000)
	catPID := libpf.PID(2000)

	// host.FileID values used in the eBPF frame data.
	goHostFileID, err := host.FileIDFromBytes(
		[]byte{0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55})
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

	// Build a libc mapping whose libpf.FileID has Hi() == libcHostFileID,
	// so findMappingForTrace can match the frame's host.FileID to this mapping.
	libcLibpfFileID := libpf.NewFileID(uint64(libcHostFileID), 0)
	libcMapping := libpf.NewFrameMapping(libpf.FrameMappingData{
		File: libpf.NewFrameMappingFile(libpf.FrameMappingFileData{
			FileID:   libcLibpfFileID,
			FileName: libpf.Intern("libc.so.6"),
		}),
		Start: 0,
		End:   0xFFFFFFF,
	})

	goODID := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 1}

	frameCache, err := lru.New[frameCacheKey, libpf.Frames](1024, hashFrameCacheKey)
	require.NoError(err)
	frameCache.SetLifetime(frameCacheLifetime)

	capture := &traceCapture{}
	pm := &ProcessManager{
		interpreters: map[libpf.PID]map[util.OnDiskFileIdentifier]interpreter.Instance{
			goPID: {goODID: goInstance},
			// catPID has NO interpreters — it's a plain C program.
		},
		pidToProcessInfo: map[libpf.PID]*processInfo{
			goPID: {mappings: []Mapping{{FrameMapping: libcMapping}}},
			catPID: {mappings: []Mapping{{FrameMapping: libcMapping}}},
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

	// Step 1: HandleTrace for the Go process. The Go interpreter will
	// incorrectly symbolize the libc frame because it doesn't check file ID.
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

	// BUG: The frame got Go type and Go function name despite being from libc.
	assert.Equal(libpf.GoFrame, goFrame.Type,
		"confirming the bug: libc frame was rewritten to GoFrame")
	assert.NotEmpty(goFrame.FunctionName.String(),
		"confirming the bug: libc frame got a Go function name")

	// Step 2: HandleTrace for the cat process with the exact same frame data.
	// Cat has no Go interpreter, but the incorrectly symbolized frame is now
	// in the cache with a PID-agnostic key (native frames don't set PIDSpecific).
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
	t.Logf("Cat process frame (from cache): type=%v func=%q",
		catFrame.Type, catFrame.FunctionName)

	// BUG: cat inherited the Go-symbolized frame from the cache.
	assert.Equal(libpf.GoFrame, catFrame.Type,
		"confirming the bug: cat process inherited GoFrame from cache")
	assert.NotEmpty(catFrame.FunctionName.String(),
		"confirming the bug: cat process inherited Go function name from cache")

	// Verify the cache was involved: the Go process should have been a miss
	// and cat should have been a hit.
	assert.Equal(uint64(1), pm.frameCacheMiss.Load(),
		"Go process frame should be a cache miss")
	assert.Equal(uint64(1), pm.frameCacheHit.Load(),
		"cat process frame should be a cache hit")
}
