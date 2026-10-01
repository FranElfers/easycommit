// Package llama is a minimal HTTP client for llama-server over a Unix socket.
package llama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
)

var (
	// ErrLoading means the server is up but still loading the model.
	ErrLoading = errors.New("llama-server is loading the model")
	// ErrTruncated means the answer did not fit in the requested token limit.
	ErrTruncated = errors.New("model answer exceeded the response token limit")
)

// temperature is low so that proposals stay close to the diff.
const temperature = 0.2

// Client talks to one llama-server instance.
type Client struct {
	http *http.Client
}

// NewClient returns a client for the server listening on the Unix socket.
func NewClient(socket string) *Client {
	return &Client{http: &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", socket)
		},
	}}}
}

// ChatRequest is a single-turn chat completion.
type ChatRequest struct {
	System    string
	User      string
	Schema    any // JSON schema constraining the answer; nil for free text
	MaxTokens int
	// Progress, when set, is called as the inference advances.
	Progress func(Progress)
}

// Progress is how far an inference is: the server first reads the prompt,
// whose size is known, and then writes the answer, whose size is not.
type Progress struct {
	PromptDone, PromptTotal int // prompt tokens read so far, and in total
	Generated               int // answer tokens written so far
}

// Health returns nil when the server is ready to answer requests.
func (c *Client) Health(ctx context.Context) error {
	return c.do(ctx, http.MethodGet, "/health", nil, nil)
}

// Sleeping reports whether the server unloaded the model after being idle.
// Asking does not wake the server up.
func (c *Client) Sleeping(ctx context.Context) (bool, error) {
	var props struct {
		IsSleeping bool `json:"is_sleeping"`
	}
	err := c.do(ctx, http.MethodGet, "/props", nil, &props)
	return props.IsSleeping, err
}

// CountTokens returns the number of tokens the loaded model sees in text.
func (c *Client) CountTokens(ctx context.Context, text string) (int, error) {
	var out struct {
		Tokens []json.RawMessage `json:"tokens"`
	}
	err := c.do(ctx, http.MethodPost, "/tokenize", map[string]any{"content": text}, &out)
	return len(out.Tokens), err
}

// Chat runs one inference and returns the text of the answer. With
// ErrTruncated the text is what the model wrote before hitting the limit.
// The answer is streamed so that req.Progress can follow it.
func (c *Client) Chat(ctx context.Context, req ChatRequest) (string, error) {
	body := map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": req.System},
			{"role": "user", "content": req.User},
		},
		"temperature":     temperature,
		"max_tokens":      req.MaxTokens,
		"stream":          true,
		"return_progress": true,
	}
	if req.Schema != nil {
		body["response_format"] = map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "answer", "strict": true, "schema": req.Schema},
		}
	}
	resp, err := c.send(ctx, http.MethodPost, "/v1/chat/completions", body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var content strings.Builder
	var progress Progress
	finish := ""
	events := bufio.NewScanner(resp.Body)
	events.Buffer(nil, 1<<20)
	for events.Scan() {
		data, ok := strings.CutPrefix(events.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Delta        struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			PromptProgress *struct {
				Total     int `json:"total"`
				Processed int `json:"processed"`
			} `json:"prompt_progress"`
			Error *apiError `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return content.String(), fmt.Errorf("llama-server sent an unreadable event: %w", err)
		}
		if chunk.Error != nil {
			return content.String(), fmt.Errorf("llama-server: %s", chunk.Error.Message)
		}
		if p := chunk.PromptProgress; p != nil {
			progress.PromptDone, progress.PromptTotal = p.Processed, p.Total
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				// The server sends one event per token.
				content.WriteString(choice.Delta.Content)
				progress.Generated++
			}
			if choice.FinishReason != "" {
				finish = choice.FinishReason
			}
		}
		if req.Progress != nil {
			req.Progress(progress)
		}
	}
	switch {
	case events.Err() != nil:
		return content.String(), events.Err()
	case finish == "":
		return content.String(), errors.New("llama-server closed the answer before finishing it")
	case finish == "length":
		return content.String(), ErrTruncated
	}
	return content.String(), nil
}

// apiError is the error object of a llama-server response.
type apiError struct {
	Message string `json:"message"`
}

// do sends a JSON request and decodes a JSON answer into out when it is not nil.
func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	resp, err := c.send(ctx, method, path, in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// send sends a JSON request and returns the response once the server has
// accepted it. The caller closes the body.
func (c *Client) send(ctx context.Context, method, path string, in any) (*http.Response, error) {
	var body io.Reader
	if in != nil {
		data, err := json.Marshal(in)
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(data)
	}
	// The host is a placeholder: the transport always dials the socket.
	req, err := http.NewRequestWithContext(ctx, method, "http://llama-server"+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusOK {
		return resp, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusServiceUnavailable {
		return nil, ErrLoading
	}
	var fail struct {
		Error apiError `json:"error"`
	}
	json.NewDecoder(resp.Body).Decode(&fail)
	return nil, fmt.Errorf("llama-server %s: %s: %s", path, resp.Status, fail.Error.Message)
}
