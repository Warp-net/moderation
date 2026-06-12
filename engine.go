//go:build llama

package moderation

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	log "github.com/sirupsen/logrus"
	llama "github.com/tcpipuk/llama-go"
)

// 	CGO_CXXFLAGS="-w -Wno-format -Wno-delete-incomplete" go run -tags=llama cmd/node/moderator/main.go --node.network testnet --node.port 4002 --node.seed moderatorlocalhost --node.moderator.modelpath Llama-Guard-3-1B.Q8_0.gguf 2>/dev/null

type Engine interface {
	Moderate(content string) (bool, string, error)
	Close()
}

type llamaEngine struct {
	model *llama.Model
	ctx   *llama.Context
	opts  llama.ChatOptions
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
	opts := llama.ChatOptions{
		MaxTokens:   llama.Int(64),
		Temperature: llama.Float32(0.0),
		TopP:        llama.Float32(0.9),
		Seed:        llama.Int(42),
	}

	lle := &llamaEngine{model: model, ctx: ctx, opts: opts}
	return lle, nil
}

func (e *llamaEngine) Moderate(content string) (bool, string, error) {
	now := time.Now()
	// Chat applies the model's embedded Llama Guard safety template (which
	// carries the hazard taxonomy); Generate/FormatChatPrompt would fall back
	// to a plain llama3 prompt and not classify.
	resp, err := e.ctx.Chat(context.Background(), []llama.ChatMessage{
		{Role: "user", Content: content},
	}, e.opts)
	if err != nil {
		return true, "", err
	}
	elapsed := time.Since(now)
	log.Infof("moderation: elapsed %s", elapsed.String())

	out := strings.ToLower(strings.TrimSpace(resp.Content))

	switch {
	case strings.HasPrefix(out, "safe"):
		return true, "", nil
	case strings.HasPrefix(out, "unsafe"):
		reason := parseViolationReason(resp.Content)
		if reason == "" {
			return true, "", nil
		}
		return false, reason, nil
	default:
		return true, "", errors.New("unrecognized LLM output: " + out)
	}
}

func (e *llamaEngine) Close() {
	_ = e.ctx.Close()
	_ = e.model.Close()
}
