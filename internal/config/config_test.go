package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoad(t *testing.T) {
	t.Run("missing file gives the defaults", func(t *testing.T) {
		c, err := Load(filepath.Join(t.TempDir(), "absent.toml"))
		if err != nil {
			t.Fatal(err)
		}
		if c.Body != "auto" || c.CtxSize != Unset || c.VRAMMB != Unset || len(c.Exclude) == 0 {
			t.Errorf("unexpected defaults: %+v", c)
		}
		if err := c.Validate(); err != nil {
			t.Errorf("defaults do not validate: %v", err)
		}
	})
	t.Run("file overrides only the keys it sets", func(t *testing.T) {
		c, err := Load(write(t, "body = \"never\"\ngpu_layers = 0\nexclude = [\"*.snap\"]\n"))
		if err != nil {
			t.Fatal(err)
		}
		if c.Body != "never" || c.GPULayers != 0 || len(c.Exclude) != 1 || c.Lang != "en" {
			t.Errorf("unexpected config: %+v", c)
		}
	})
	t.Run("unknown key is rejected", func(t *testing.T) {
		_, err := Load(write(t, "ctx_sise = 8192\n"))
		if err == nil || !strings.Contains(err.Error(), "ctx_sise") {
			t.Errorf("want an error naming the key, got %v", err)
		}
	})
	t.Run("invalid TOML is rejected", func(t *testing.T) {
		if _, err := Load(write(t, "body = \n")); err == nil {
			t.Error("want an error")
		}
	})
}

func TestValidate(t *testing.T) {
	for name, mutate := range map[string]func(*Config){
		"body":     func(c *Config) { c.Body = "sometimes" },
		"ctx size": func(c *Config) { c.CtxSize = MinCtxSize - 1 },
		"layers":   func(c *Config) { c.GPULayers = -2 },
		"lang":     func(c *Config) { c.Lang = "" },
		"pattern":  func(c *Config) { c.Exclude = []string{"[a-"} },
	} {
		c := Default()
		mutate(&c)
		if c.Validate() == nil {
			t.Errorf("%s: invalid value accepted", name)
		}
	}
}

func TestSaveVRAM(t *testing.T) {
	// The file is created with its directory, and existing content is kept
	// even when it lacks a final newline.
	path := filepath.Join(t.TempDir(), "new", "config.toml")
	if err := SaveVRAM(path, 8192); err != nil {
		t.Fatal(err)
	}
	if c, err := Load(path); err != nil || c.VRAMMB != 8192 {
		t.Errorf("created file: vram %d, err %v", c.VRAMMB, err)
	}

	path = write(t, "# my settings\nlang = \"es\"")
	if err := SaveVRAM(path, 0); err != nil {
		t.Fatal(err)
	}
	c, err := Load(path)
	if err != nil || c.VRAMMB != 0 || c.Lang != "es" {
		t.Errorf("appended file: %+v, err %v", c, err)
	}
	if data, _ := os.ReadFile(path); !strings.HasPrefix(string(data), "# my settings\n") {
		t.Errorf("existing content was not preserved: %q", data)
	}
}

func TestWithProfile(t *testing.T) {
	tests := []struct {
		name               string
		vram, ctx, layers  int
		cpu                bool
		wantCtx, wantLayer int
		wantGPU            bool
	}{
		{"16 GB card", 16304, Unset, Unset, false, 32768, AllLayers, true},
		{"12 GB card", 12288, Unset, Unset, false, 16384, AllLayers, true},
		{"8 GB card", 8192, Unset, Unset, false, 8192, AllLayers, true},
		{"6 GB card offloads part of the model", 6144, Unset, Unset, false, 8192, 24, true},
		{"4 GB card", 4096, Unset, Unset, false, 8192, 12, true},
		{"too little VRAM keeps the model on the CPU", 2048, Unset, Unset, false, 8192, 0, true},
		{"no GPU", 0, Unset, Unset, false, 8192, 0, false},
		{"explicit values win over the profile", 16304, 4096, 10, false, 4096, 10, true},
		{"cpu flag ignores VRAM and layers", 16304, Unset, 10, true, 8192, 0, false},
		{"no GPU ignores explicit layers", 0, 16384, 10, false, 16384, 0, false},
	}
	for _, tt := range tests {
		c := Default()
		c.VRAMMB, c.CtxSize, c.GPULayers, c.CPU = tt.vram, tt.ctx, tt.layers, tt.cpu
		got := c.WithProfile()
		if got.CtxSize != tt.wantCtx || got.GPULayers != tt.wantLayer || got.GPU() != tt.wantGPU {
			t.Errorf("%s: ctx %d, layers %d, gpu %v; want %d, %d, %v",
				tt.name, got.CtxSize, got.GPULayers, got.GPU(), tt.wantCtx, tt.wantLayer, tt.wantGPU)
		}
	}
}
