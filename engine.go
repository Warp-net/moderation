//go:build llama

package moderation

import (
	"errors"
	"fmt"
	"os"
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

type llamaEngine struct {
	model *llama.Model
	ctx   *llama.Context
	opts  []llama.GenerateOption
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

	lle := &llamaEngine{model: model, ctx: ctx, opts: opts}
	return lle, nil
}

func (e *llamaEngine) Moderate(content string) (bool, string, error) {
	now := time.Now()
	raw, err := e.ctx.Generate(buildGuardPrompt(content), e.opts...)
	if err != nil {
		return true, "", err
	}
	elapsed := time.Since(now)
	log.Infof("moderation: elapsed %s", elapsed.String())

	return classifyOutput(raw)
}

func (e *llamaEngine) Close() {
	_ = e.ctx.Close()
	_ = e.model.Close()
}
