package server

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const repoFlag = "--hf-repo"

// modelArgs returns the llama-server arguments that select model, which is
// either a path to a .gguf file or a Hugging Face repository
// ("org/name[:quant]"). llama-server looks a repository up in its own cache
// and downloads it when it is missing.
func modelArgs(model string) []string {
	if strings.HasSuffix(model, ".gguf") {
		if rest, ok := strings.CutPrefix(model, "~/"); ok {
			if home, err := os.UserHomeDir(); err == nil {
				model = filepath.Join(home, rest)
			}
		}
		return []string{"--model", model}
	}
	// The multimodal projector is never needed to read diffs.
	return []string{repoFlag, model, "--no-mmproj"}
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
