//go:build llama

package moderation

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
	llama "github.com/tcpipuk/llama-go"
)

func init() {
	// Silence llama.cpp's chatty C++ logs (the model-loader metadata dump,
	// print_info, load_tensors, sched_reserve, and the benign n_ctx warning).
	// Real load/inference failures still surface as Go errors. Set before any
	// model is loaded so it takes effect.
	_ = os.Setenv("LLAMA_LOG", "error")
	llama.InitLogging()
}

type Engine interface {
	Moderate(content string) (bool, string, error)
	Close()
}

var errEngineClosed = errors.New("moderation engine is closed")

type llamaEngine struct {
	model *llama.Model
	ctx   *llama.Context
	opts  []llama.GenerateOption

	jobs      chan generateJob
	closed    chan struct{}
	done      chan struct{}
	closeOnce sync.Once
}

type generateJob struct {
	prompt string
	result chan generateResult
}

type generateResult struct {
	raw string
	err error
}

func NewLlamaEngine(modelPath string, threads int) (_ *llamaEngine, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = errors.New(fmt.Sprint(r))
		}
	}()
	if modelPath == "" {
		return nil, errors.New("model path is required")
	}
	model, err := llama.LoadModel(
		modelPath,
		llama.WithMMap(true),
		llama.WithGPULayers(0),
		llama.WithSilentLoading(),
	)
	if err != nil {
		return nil, err
	}

	ctx, err := model.NewContext(
		llama.WithContext(4096),
		llama.WithBatch(256),
		llama.WithThreads(threads),
	)
	if err != nil {
		model.Close()
		return nil, err
	}

	// Llama Guard classifies, so we only need a few tokens ("unsafe\nS9,S2")
	// and deterministic decoding.
	opts := []llama.GenerateOption{
		llama.WithMaxTokens(64),
		llama.WithTemperature(0.0),
		llama.WithTopP(0.9),
		llama.WithSeed(42),
	}

	lle := &llamaEngine{
		model:  model,
		ctx:    ctx,
		opts:   opts,
		jobs:   make(chan generateJob),
		closed: make(chan struct{}),
		done:   make(chan struct{}),
	}
	go lle.serve()
	return lle, nil
}

func (e *llamaEngine) serve() {
	runtime.LockOSThread()
	defer close(e.done)
	for {
		select {
		case job := <-e.jobs:
			raw, err := e.ctx.Generate(job.prompt, e.opts...)
			job.result <- generateResult{raw: raw, err: err}
		case <-e.closed:
			return
		}
	}
}

func (e *llamaEngine) generate(prompt string) (string, error) {
	job := generateJob{prompt: prompt, result: make(chan generateResult, 1)}
	select {
	case e.jobs <- job:
	case <-e.closed:
		return "", errEngineClosed
	}
	res := <-job.result
	return res.raw, res.err
}

func (e *llamaEngine) Moderate(content string) (bool, string, error) {
	now := time.Now()
	raw, err := e.generate(buildGuardPrompt(content))
	if err != nil {
		return true, "", err
	}
	elapsed := time.Since(now)
	log.Infof("moderation: elapsed %s", elapsed.String())

	return classifyOutput(raw)
}

func (e *llamaEngine) Close() {
	e.closeOnce.Do(func() {
		close(e.closed)
		<-e.done
		_ = e.ctx.Close()
		_ = e.model.Close()
	})
}
