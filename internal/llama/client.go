// Package llama is a minimal HTTP client for llama-server over a Unix socket.
package llama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
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
func (c *Client) Chat(ctx context.Context, req ChatRequest) (string, error) {
	body := map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": req.System},
			{"role": "user", "content": req.User},
		},
		"temperature": temperature,
		"max_tokens":  req.MaxTokens,
	}
	if req.Schema != nil {
		body["response_format"] = map[string]any{
			"type":        "json_schema",
			"json_schema": map[string]any{"name": "answer", "strict": true, "schema": req.Schema},
		}
	}
	var out struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := c.do(ctx, http.MethodPost, "/v1/chat/completions", body, &out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", errors.New("llama-server returned no choices")
	}
	content := out.Choices[0].Message.Content
	if out.Choices[0].FinishReason == "length" {
		return content, ErrTruncated
	}
	return content, nil
}

// do sends a JSON request and decodes a JSON answer into out when it is not nil.
func (c *Client) do(ctx context.Context, method, path string, in, out any) error {
	var body io.Reader
	if in != nil {
		data, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(data)
	}
	// The host is a placeholder: the transport always dials the socket.
	req, err := http.NewRequestWithContext(ctx, method, "http://llama-server"+path, body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode == http.StatusServiceUnavailable {
		return ErrLoading
	}
	if resp.StatusCode != http.StatusOK {
		var fail struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.Unmarshal(data, &fail)
		return fmt.Errorf("llama-server %s: %s: %s", path, resp.Status, fail.Error.Message)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}
