// Package config holds easycommit settings: defaults, the TOML file and the
// VRAM profiles that fill in whatever the user left unset.
package config

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/BurntSushi/toml"
)

const (
	// Unset marks a value to be resolved from the VRAM profile or by autodetection.
	Unset = -1
	// AllLayers exceeds the layer count of any model, so llama.cpp offloads all of them.
	AllLayers = 999
	// MinCtxSize leaves room for the prompt scaffolding and the response reserve.
	MinCtxSize = 4096
)

// BodyModes are the accepted values of Config.Body.
var BodyModes = []string{"auto", "never", "always"}

// Config mirrors the TOML file; command-line flags override its fields.
type Config struct {
	ServerBin   string   `toml:"server_bin"`
	Model       string   `toml:"model"`
	CtxSize     int      `toml:"ctx_size"`
	GPULayers   int      `toml:"gpu_layers"`
	CPU         bool     `toml:"cpu"`
	Body        string   `toml:"body"`
	Lang        string   `toml:"lang"`
	IdleTimeout int      `toml:"idle_timeout"`
	Exclude     []string `toml:"exclude"`
	VRAMMB      int      `toml:"vram_mb"`
}

// Default returns the configuration used when neither file nor flags say otherwise.
func Default() Config {
	return Config{
		ServerBin:   "llama-server",
		Model:       "mistralai/Ministral-3-8B-Instruct-2512-GGUF:Q4_K_M",
		CtxSize:     Unset,
		GPULayers:   Unset,
		Body:        "auto",
		Lang:        "en",
		IdleTimeout: 600,
		Exclude: []string{
			"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "go.sum",
			"Cargo.lock", "composer.lock", "poetry.lock", "uv.lock",
			"vendor/", "node_modules/", "dist/", "build/",
			"*.min.js", "*.min.css", "*.map",
		},
		VRAMMB: Unset,
	}
}

// DefaultPath is the configuration file location under the user config directory.
func DefaultPath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "easycommit", "config.toml"), nil
}

// Load overlays the file at path on the defaults. A missing file is not an error.
func Load(path string) (Config, error) {
	c := Default()
	md, err := toml.DecodeFile(path, &c)
	if errors.Is(err, fs.ErrNotExist) {
		return c, nil
	}
	if err != nil {
		return c, fmt.Errorf("config %s: %w", path, err)
	}
	if unknown := md.Undecoded(); len(unknown) > 0 {
		return c, fmt.Errorf("config %s: unknown key %q", path, unknown[0])
	}
	return c, nil
}

// Validate rejects values that no part of the program can work with.
func (c Config) Validate() error {
	if !slices.Contains(BodyModes, c.Body) {
		return fmt.Errorf("body must be one of %s, got %q", strings.Join(BodyModes, ", "), c.Body)
	}
	if c.CtxSize != Unset && c.CtxSize < MinCtxSize {
		return fmt.Errorf("ctx_size must be at least %d, got %d", MinCtxSize, c.CtxSize)
	}
	if c.GPULayers < Unset {
		return fmt.Errorf("gpu_layers must not be negative, got %d", c.GPULayers)
	}
	if c.Lang == "" || c.Model == "" || c.ServerBin == "" {
		return errors.New("lang, model and server_bin must not be empty")
	}
	for _, p := range c.Exclude {
		if _, err := path.Match(p, ""); err != nil {
			return fmt.Errorf("exclude pattern %q: %w", p, err)
		}
	}
	return nil
}

// SaveVRAM records the detected VRAM by appending the key to the file at path,
// creating it if needed and leaving the rest of its content untouched.
func SaveVRAM(path string, mb int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	old, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	if len(old) > 0 && !strings.HasSuffix(string(old), "\n") {
		old = append(old, '\n')
	}
	return os.WriteFile(path, fmt.Appendf(old, "vram_mb = %d\n", mb), 0o644)
}
