//go:build llama

package moderation

import (
	"os"
	"runtime"
	"sync"
	"testing"
	"time"
)

const modelPath = "Llama-Guard-3-1B.Q8_0.gguf"

func TestNewLlamaEngineViolationSuccess(t *testing.T) {
	if testing.Short() {
		t.Skip()
		return
	}
	if runtime.GOOS != "linux" {
		t.Skip()
		return
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip()
		return
	}

	eng, err := NewLlamaEngine(modelPath, runtime.NumCPU())
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	now := time.Now()
	ok, reason, err := eng.Moderate("I'm selling AK-47 and cocaine")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("elapsed: %s", time.Since(now))

	t.Logf("result: ok: %t, reason: %s", ok, reason)

	if ok {
		t.Fatal("moderation must fail", reason)
	}
}

func TestNewLlamaEngineNoViolationSuccess(t *testing.T) {
	if testing.Short() {
		t.Skip()
		return
	}
	if runtime.GOOS != "linux" {
		t.Skip()
		return
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip()
		return
	}

	eng, err := NewLlamaEngine(modelPath, runtime.NumCPU())
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	now := time.Now()
	ok, reason, err := eng.Moderate("I'm selling automatic transmission and moccasin.")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("elapsed: %s", time.Since(now))

	t.Logf("result: ok: %t, reason: %s", ok, reason)

	if !ok {
		t.Fatal("moderation must succeed:", reason)
	}
}

func TestModerateFromManyThreadsKeepsOneThreadTeam(t *testing.T) {
	if testing.Short() {
		t.Skip()
		return
	}
	if runtime.GOOS != "linux" {
		t.Skip()
		return
	}
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skip()
		return
	}

	const threads, callers = 12, 4
	eng, err := NewLlamaEngine(modelPath, threads)
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	if _, _, err := eng.Moderate("Good morning everyone."); err != nil {
		t.Fatal(err)
	}
	before := osThreads(t)

	var moderated sync.WaitGroup
	release := make(chan struct{})
	moderated.Add(callers)
	for range callers {
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			if _, _, err := eng.Moderate("Good morning everyone."); err != nil {
				t.Error(err)
			}
			moderated.Done()
			<-release
		}()
	}
	moderated.Wait()
	grown := osThreads(t) - before
	close(release)

	if grown >= threads-1 {
		t.Fatalf("moderating from %d threads started %d new ones", callers, grown)
	}
}

func osThreads(t *testing.T) int {
	t.Helper()
	tasks, err := os.ReadDir("/proc/self/task")
	if err != nil {
		t.Fatal(err)
	}
	return len(tasks)
}
