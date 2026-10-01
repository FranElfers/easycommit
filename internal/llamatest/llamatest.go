// Package llamatest provides a fake llama-server for tests: the same command
// line and HTTP API as the real one, with deterministic answers and no model.
package llamatest

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

const (
	// fakeEnv makes Main take over the test binary.
	fakeEnv = "EASYCOMMIT_FAKE_LLAMA_SERVER"
	// DevicesEnv overrides the output of --list-devices.
	DevicesEnv = "LLAMATEST_DEVICES"
	// FailEnv makes the fake server exit at startup with this message.
	FailEnv = "LLAMATEST_FAIL"

	// OneGPU is the default device list; NoGPU is what a host without one prints.
	OneGPU = "Available devices:\n  Vulkan0: Fake GPU (16304 MiB, 16000 MiB free)\n"
	NoGPU  = "Available devices:\n"
	// GroupName is the single group every fake proposal has.
	GroupName = "all changes"
)

// Main runs the fake server and exits when the test binary was started
// through the script made by Install. Call it first in TestMain.
func Main() {
	if os.Getenv(fakeEnv) == "" {
		return
	}
	args := os.Args[1:]
	if slices.Contains(args, "--list-devices") {
		devices, ok := os.LookupEnv(DevicesEnv)
		if !ok {
			devices = OneGPU
		}
		fmt.Print(devices)
		os.Exit(0)
	}
	if msg := os.Getenv(FailEnv); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
		os.Exit(1)
	}
	if slices.Contains(args, "--hf-repo") {
		fmt.Fprintln(os.Stderr, "fake download progress")
	}
	listener, err := net.Listen("unix", args[slices.Index(args, "--host")+1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, http.Serve(listener, Handler()))
	os.Exit(1)
}

// Install writes an executable llama-server that runs the current test
// binary as the fake server, and returns its path.
func Install(t testing.TB) string {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf("#!/bin/sh\nexec env %s=1 %q \"$@\"\n", fakeEnv, self)
	path := filepath.Join(t.TempDir(), "llama-server")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// ShortDir returns a temporary directory whose path is short enough to hold
// a Unix socket, which t.TempDir does not guarantee.
func ShortDir(t testing.TB) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "ec")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}

// Handler serves the subset of the llama-server API that easycommit uses.
func Handler() http.Handler {
	mux := http.NewServeMux()
	reply := func(w http.ResponseWriter, v any) { json.NewEncoder(w).Encode(v) }
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		reply(w, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /props", func(w http.ResponseWriter, _ *http.Request) {
		reply(w, map[string]bool{"is_sleeping": false})
	})
	mux.HandleFunc("POST /tokenize", func(w http.ResponseWriter, r *http.Request) {
		var in struct{ Content string }
		json.NewDecoder(r.Body).Decode(&in)
		reply(w, map[string][]int{"tokens": make([]int, CountTokens(in.Content))})
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			ResponseFormat struct {
				JSONSchema struct{ Schema json.RawMessage } `json:"json_schema"`
			} `json:"response_format"`
		}
		json.NewDecoder(r.Body).Decode(&in)
		reply(w, map[string]any{"choices": []any{map[string]any{
			"finish_reason": "stop",
			"message":       map[string]string{"content": Answer(in.ResponseFormat.JSONSchema.Schema)},
		}}})
	})
	return mux
}

// CountTokens is the fake tokenizer: one token per whitespace-separated word.
func CountTokens(text string) int {
	return len(strings.Fields(text))
}

// Answer returns what the fake model says under a response schema: a valid
// proposal with every file in one group, a summary per file, or free text
// when there is no schema.
func Answer(schema json.RawMessage) string {
	type object struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	var top object
	if json.Unmarshal(schema, &top); len(top.Properties) == 0 {
		return "summary of the whole file"
	}

	filesSchema, isProposal := top.Properties["files"]
	if !isProposal {
		summaries := map[string]string{}
		for path := range top.Properties {
			summaries[path] = "summary of " + path
		}
		return marshal(summaries)
	}

	var files, message object
	json.Unmarshal(filesSchema, &files)
	json.Unmarshal(top.Properties["all_in_one"], &message)
	assigned := map[string]string{}
	for path := range files.Properties {
		assigned[path] = GroupName
	}
	msg := map[string]any{"subject": fmt.Sprintf("chore: update %d files", len(assigned))}
	if bodySchema, ok := message.Properties["body"]; ok {
		msg["body"] = nil
		if !strings.Contains(string(bodySchema), "null") {
			msg["body"] = "Because the tests say so."
		}
	}
	group := map[string]any{"name": GroupName}
	for k, v := range msg {
		group[k] = v
	}
	return marshal(map[string]any{
		"plan": []string{GroupName}, "files": assigned, "groups": []any{group}, "all_in_one": msg,
	})
}

func marshal(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return string(data)
}
