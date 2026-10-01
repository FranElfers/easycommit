package server

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	repoFlag = "--hf-repo"
	// defaultQuant is what llama-server picks when the repository name has no tag.
	defaultQuant = "Q4_K_M"
)

// modelArgs returns the llama-server arguments that can select model, in
// order of preference. model is either a path to a .gguf file or a Hugging
// Face repository ("org/name[:quant]"). A repository already in the local
// cache is loaded from its file, which needs no network access; the
// repository form is what makes llama-server download it.
func modelArgs(model string) [][]string {
	if strings.HasSuffix(model, ".gguf") {
		if rest, ok := strings.CutPrefix(model, "~/"); ok {
			if home, err := os.UserHomeDir(); err == nil {
				model = filepath.Join(home, rest)
			}
		}
		return [][]string{{"--model", model}}
	}
	// The multimodal projector is never needed to read diffs.
	repo := []string{repoFlag, model, "--no-mmproj"}
	if file := cachedModel(model); file != "" {
		return [][]string{{"--model", file}, repo}
	}
	return [][]string{repo}
}

// cachedModel finds the GGUF file of a repository in the Hugging Face cache,
// which is where llama-server stores its downloads. It returns "" if absent.
func cachedModel(repo string) string {
	name, quant, ok := strings.Cut(repo, ":")
	if !ok {
		quant = defaultQuant
	}
	pattern := filepath.Join(hubDir(), "models--"+strings.ReplaceAll(name, "/", "--"), "snapshots", "*", "*.gguf")
	files, _ := filepath.Glob(pattern)
	slices.Sort(files) // the first part of a split model sorts first
	for _, f := range files {
		base := strings.ToLower(filepath.Base(f))
		if strings.Contains(base, strings.ToLower(quant)) && !strings.Contains(base, "mmproj") {
			return f
		}
	}
	return ""
}

// hubDir resolves the Hugging Face cache like llama.cpp and huggingface_hub do.
func hubDir() string {
	for _, env := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"} {
		if dir := os.Getenv(env); dir != "" {
			return dir
		}
	}
	if home := os.Getenv("HF_HOME"); home != "" {
		return filepath.Join(home, "hub")
	}
	cache, _ := os.UserCacheDir()
	return filepath.Join(cache, "huggingface", "hub")
}

// deviceLine matches an entry of `llama-server --list-devices`, such as
// "  Vulkan0: AMD Radeon RX 9060 XT (16304 MiB, 15210 MiB free)".
var deviceLine = regexp.MustCompile(`(?m)^\s+\S+: .*\((\d+) MiB, \d+ MiB free\)\s*$`)

// DetectVRAM returns the total memory in MiB of the largest GPU that the
// llama-server binary can use, or 0 when it reports none.
func DetectVRAM(ctx context.Context, bin string) (int, error) {
	out, err := exec.CommandContext(ctx, bin, "--list-devices").CombinedOutput()
	if err != nil {
		return 0, err
	}
	best := 0
	for _, m := range deviceLine.FindAllSubmatch(out, -1) {
		mb, _ := strconv.Atoi(string(m[1]))
		best = max(best, mb)
	}
	return best, nil
}
