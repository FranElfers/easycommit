package config

// Profile is the inference setup chosen for a range of VRAM sizes.
type Profile struct {
	MinVRAMMB int
	CtxSize   int
	GPULayers int
}

// profiles is ordered from most to least VRAM; the last entry is the CPU fallback.
// Values are provisional: see docs/development.md for the measurements behind them.
var profiles = []Profile{
	{MinVRAMMB: 14000, CtxSize: 32768, GPULayers: AllLayers},
	{MinVRAMMB: 10000, CtxSize: 16384, GPULayers: AllLayers},
	{MinVRAMMB: 7500, CtxSize: 8192, GPULayers: AllLayers},
	{MinVRAMMB: 5500, CtxSize: 8192, GPULayers: 24},
	{MinVRAMMB: 3500, CtxSize: 8192, GPULayers: 12},
	{MinVRAMMB: 0, CtxSize: 8192, GPULayers: 0},
}

// ProfileFor returns the first profile whose VRAM requirement is met.
func ProfileFor(vramMB int) Profile {
	for _, p := range profiles {
		if vramMB >= p.MinVRAMMB {
			return p
		}
	}
	return profiles[len(profiles)-1]
}

// GPU reports whether inference may use a GPU at all.
func (c Config) GPU() bool {
	return !c.CPU && c.VRAMMB > 0
}

// WithProfile fills the unset context size and GPU layers from the VRAM profile.
// Without a usable GPU every layer stays on the CPU, whatever the user asked for.
func (c Config) WithProfile() Config {
	vram := 0
	if c.GPU() {
		vram = c.VRAMMB
	}
	p := ProfileFor(vram)
	if c.CtxSize == Unset {
		c.CtxSize = p.CtxSize
	}
	if c.GPULayers == Unset || !c.GPU() {
		c.GPULayers = p.GPULayers
	}
	return c
}
