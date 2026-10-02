package llama_test

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/franelfers/easycommit/internal/llama"
	"github.com/franelfers/easycommit/internal/llamatest"
)

// serve runs handler on a Unix socket and returns a client connected to it.
func serve(t *testing.T, handler http.Handler) *llama.Client {
	t.Helper()
	socket := filepath.Join(llamatest.ShortDir(t), "server.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	go http.Serve(listener, handler)
	t.Cleanup(func() { listener.Close() })
	return llama.NewClient(socket)
}

func TestClient(t *testing.T) {
	ctx := context.Background()
	c := serve(t, llamatest.Handler())

	if err := c.Health(ctx); err != nil {
		t.Errorf("Health: %v", err)
	}
	if sleeping, err := c.Sleeping(ctx); err != nil || sleeping {
		t.Errorf("Sleeping: %v, %v", sleeping, err)
	}
	if n, err := c.CountTokens(ctx, "three short words"); err != nil || n != 3 {
		t.Errorf("CountTokens: %d, %v", n, err)
	}
	// The answer arrives in pieces, with the progress reported along the way.
	var seen []llama.Progress
	req := llama.ChatRequest{System: "s", User: "u", MaxTokens: 10, Progress: func(p llama.Progress) { seen = append(seen, p) }}
	text, err := c.Chat(ctx, req)
	if err != nil || text != llamatest.Answer(nil) {
		t.Errorf("Chat: %q, %v", text, err)
	}
	total := llamatest.PromptTokens
	if len(seen) < 4 || seen[0] != (llama.Progress{PromptDone: total / 2, PromptTotal: total}) ||
		seen[len(seen)-1] != (llama.Progress{PromptDone: total, PromptTotal: total, Generated: 2}) {
		t.Errorf("progress: %+v", seen)
	}
}

func TestChatRequest(t *testing.T) {
	var got struct {
		Messages       []struct{ Role, Content string }
		MaxTokens      int  `json:"max_tokens"`
		Stream         bool `json:"stream"`
		ReturnProgress bool `json:"return_progress"`
		ResponseFormat struct {
			Type       string
			JSONSchema struct{ Schema map[string]any } `json:"json_schema"`
		} `json:"response_format"`
	}
	finish := "stop"
	c := serve(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&got)
		llamatest.Stream(w, "partial", finish)
	}))

	req := llama.ChatRequest{System: "sys", User: "usr", MaxTokens: 42, Schema: map[string]string{"type": "object"}}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 2 || got.Messages[0].Role != "system" || got.Messages[1].Content != "usr" {
		t.Errorf("messages: %+v", got.Messages)
	}
	if got.MaxTokens != 42 || !got.Stream || !got.ReturnProgress ||
		got.ResponseFormat.Type != "json_schema" || got.ResponseFormat.JSONSchema.Schema["type"] != "object" {
		t.Errorf("request: %+v", got)
	}

	// An answer cut at the token limit is returned together with the error.
	finish = "length"
	text, err := c.Chat(context.Background(), req)
	if !errors.Is(err, llama.ErrTruncated) || text != "partial" {
		t.Errorf("truncated answer: %q, %v", text, err)
	}
}

func TestErrors(t *testing.T) {
	status := http.StatusServiceUnavailable
	c := serve(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		w.Write([]byte(`{"error":{"message":"the request exceeds the available context size"}}`))
	}))
	if err := c.Health(context.Background()); err == nil {
		t.Error("503: want an error")
	}
	status = http.StatusBadRequest
	_, err := c.Chat(context.Background(), llama.ChatRequest{})
	if err == nil || !strings.Contains(err.Error(), "exceeds the available context size") {
		t.Errorf("400: want the server message, got %v", err)
	}
	if err := llama.NewClient("/nonexistent/server.sock").Health(context.Background()); err == nil {
		t.Error("missing socket: want an error")
	}

	// Failures after the answer started: an error event, and a stream that
	// ends without a finish reason.
	for name, events := range map[string]string{
		"error event": "data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\ndata: {\"error\":{\"message\":\"slot failed\"}}\n\n",
		"cut stream":  "data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n\n",
	} {
		c := serve(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(events)) }))
		if text, err := c.Chat(context.Background(), llama.ChatRequest{}); err == nil || errors.Is(err, llama.ErrTruncated) || text != "par" {
			t.Errorf("%s: %q, %v", name, text, err)
		}
	}
}
