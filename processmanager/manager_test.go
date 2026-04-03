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

// TestFrameCacheCrossProcessPollution demonstrates that the frame cache
// serves incorrectly symbolized frames to unrelated processes because
// native frames use PID-agnostic cache keys.
func TestFrameCacheCrossProcessPollution(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)

	exec, err := os.Executable()
	require.NoError(err)

	pc, _, _, ok := runtime.Caller(0)
	require.True(ok)

	goPID := libpf.PID(1000)
	catPID := libpf.PID(2000)

	goFileID, err := host.FileIDFromBytes([]byte{0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55})
	require.NoError(err)

	libcFileID, err := host.FileIDFromBytes([]byte{0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE})
	require.NoError(err)

	// Load the Go interpreter from the test binary.
	realPID := libpf.PID(os.Getpid())
	pid := process.New(realPID, realPID)
	elfRef := pfelf.NewReference(exec, pid)
	loaderInfo := interpreter.NewLoaderInfo(goFileID, elfRef)
	rm := remotememory.NewProcessVirtualMemory(realPID)

	goData, err := golang.Loader(nil, loaderInfo)
	require.NoError(err)
	goInstance, err := goData.Attach(nil, realPID, 0x0, rm)
	require.NoError(err)

	// Build FrameMapping for libc — this is what findMappingForTrace would return.
	libcMapping := libpf.NewFrameMapping(libpf.FrameMappingData{
		File: libpf.NewFrameMappingFile(libpf.FrameMappingFileData{
			FileName: libpf.Intern("libc.so.6"),
		}),
		Start: 0,
		End:   0xFFFFFFF,
	})

	goODID := util.OnDiskFileIdentifier{DeviceID: 1, InodeNum: 1}

	frameCache, err := lru.New[frameCacheKey, libpf.Frames](1024, hashFrameCacheKey)
	require.NoError(err)
	frameCache.SetLifetime(frameCacheLifetime)

	pm := &ProcessManager{
		interpreters: map[libpf.PID]map[util.OnDiskFileIdentifier]interpreter.Instance{
			goPID: {goODID: goInstance},
			// catPID has NO interpreters (plain C program)
		},
		pidToProcessInfo: map[libpf.PID]*processInfo{
			goPID: {
				mappings: []Mapping{{
					FrameMapping: libcMapping,
				}},
			},
			catPID: {
				mappings: []Mapping{{
					FrameMapping: libcMapping,
				}},
			},
		},
		frameCache: frameCache,
	}

	// Construct a native frame at a Go function's address but with libc's file ID.
	// This simulates a libc frame whose file offset collides with a Go pclntab entry.
	ef := libpf.NewEbpfFrame(libpf.NativeFrame, 0, 2, uint64(pc))
	ef[1] = uint64(libcFileID)

	// Step 1: Process the frame for the Go process. The Go interpreter will
	// incorrectly symbolize it because it doesn't check the file ID.
	var goFrames libpf.Frames
	cached := pm.convertFrame(goPID, ef, &goFrames)

	if !cached {
		t.Log("convertFrame returned false (not cacheable) — bug may be fixed")
		return
	}

	// The Go interpreter claimed the frame. It should have been left as a
	// plain native frame with the libc mapping.
	require.NotEmpty(goFrames)
	goFrame := goFrames[0].Value()

	t.Logf("Go process frame: type=%v func=%q", goFrame.Type, goFrame.FunctionName)

	// BUG: The frame got Go type and Go function name despite being from libc.
	assert.Equal(libpf.GoFrame, goFrame.Type,
		"confirming the bug: frame was rewritten to GoFrame")
	assert.NotEmpty(goFrame.FunctionName.String(),
		"confirming the bug: frame got a Go function name")

	// Step 2: Cache the frame as HandleTrace would (native frames have pid=0 in key).
	key := frameCacheKey{}
	// Native frames don't have PIDSpecific flag, so key.pid stays 0.
	copy(key.data[:], ef)
	pm.frameCache.Add(key, goFrames)

	// Step 3: The cat process looks up the same frame — cache hit.
	cachedFrames, hit := pm.frameCache.GetAndRefresh(key, frameCacheLifetime)

	// BUG: cat gets the Go-symbolized frame from cache.
	require.True(hit, "expected cache hit for cat process")
	require.NotEmpty(cachedFrames)

	catFrame := cachedFrames[0].Value()
	t.Logf("Cat process frame (from cache): type=%v func=%q", catFrame.Type, catFrame.FunctionName)

	// A plain C program should never have GoFrame type or Go function names.
	assert.Equal(libpf.GoFrame, catFrame.Type,
		"confirming the bug: cat process inherited GoFrame from cache")
	assert.NotEmpty(catFrame.FunctionName.String(),
		"confirming the bug: cat process inherited Go function name from cache")
}
