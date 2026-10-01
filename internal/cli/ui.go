package cli

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/franelfers/easycommit/internal/propose"
)

// refreshInterval is how often the live line is redrawn. It paces the
// elapsed-time counter and caps how often progress reaches the screen: the
// server reports once per token, far more often than is worth drawing.
const refreshInterval = 100 * time.Millisecond

const (
	// barWidth is the number of cells of the progress bar.
	barWidth = 20
	// liveWidth caps the live line so that it never wraps, which would
	// break rewriting it in place.
	liveWidth = 79
)

// ui is everything easycommit writes to stderr. On a terminal it keeps one
// live line with the current progress, which it erases before printing
// anything else and when the work ends. Elsewhere it prints plain lines.
type ui struct {
	mu      sync.Mutex
	w       io.Writer
	tty     bool
	refresh time.Duration // see refreshInterval

	live  bool      // the live line is on screen
	phase string    // what the live line is showing
	drawn time.Time // when the live line was last drawn
	busy  bool      // something was printed since the last refresh
	title string    // last step announced when not on a terminal
}

func newUI(w io.Writer) *ui {
	u := &ui{w: w, refresh: refreshInterval}
	if f, ok := w.(*os.File); ok {
		info, err := f.Stat()
		u.tty = err == nil && info.Mode()&os.ModeCharDevice != 0
	}
	return u
}

// Write prints output produced elsewhere, such as the server log.
func (u *ui) Write(p []byte) (int, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.erase()
	u.busy = true
	return u.w.Write(p)
}

// note prints a message on a line of its own.
func (u *ui) note(format string, args ...any) {
	fmt.Fprintf(u, "easycommit: "+format+"\n", args...)
}

// clear removes the live line.
func (u *ui) clear() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.erase()
}

func (u *ui) erase() {
	if u.live {
		io.WriteString(u.w, "\r\x1b[K")
		u.live = false
	}
}

// show replaces the live line. The caller holds the lock.
func (u *ui) show(line string) {
	if r := []rune(line); len(r) > liveWidth {
		line = string(r[:liveWidth-1]) + "…"
	}
	fmt.Fprintf(u.w, "\r%s\x1b[K", line)
	u.live, u.drawn = true, time.Now()
}

// status reports the inference in progress: a bar while the model reads the
// prompt, whose size is known, and a token count while it writes the answer.
// Off a terminal it prints one line when each step starts.
func (u *ui) status(s propose.Status) {
	title := s.Label
	if s.Steps > 1 {
		title = fmt.Sprintf("step %d of %d: %s", s.Step, s.Steps, s.Label)
	}
	if !u.tty {
		if title != u.title {
			u.title = title
			u.note("%s", title)
		}
		return
	}
	line, phase := title, "start"
	switch {
	case s.Generated > 0:
		line, phase = fmt.Sprintf("writing answer, %d tokens (%s)", s.Generated, title), "write"
	case s.PromptTotal > 0:
		cells := s.PromptDone * barWidth / s.PromptTotal
		bar := strings.Repeat("#", cells) + strings.Repeat("-", barWidth-cells)
		line, phase = fmt.Sprintf("[%s] %3d%% reading changes (%s)", bar, s.PromptDone*100/s.PromptTotal, title), "read"
	}
	phase += title

	u.mu.Lock()
	defer u.mu.Unlock()
	// A new step or phase shows at once; updates within one wait their turn.
	if u.live && phase == u.phase && time.Since(u.drawn) < u.refresh {
		return
	}
	u.phase = phase
	u.show(line)
}

// wait runs f. On a terminal, while f runs and prints nothing itself, the
// live line shows label and the seconds elapsed.
func (u *ui) wait(label string, f func() error) error {
	if !u.tty {
		return f()
	}
	stop, stopped := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(stopped)
		start := time.Now()
		ticker := time.NewTicker(u.refresh)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				u.mu.Lock()
				if u.busy {
					u.busy = false
				} else {
					u.show(fmt.Sprintf("%s… %d s", label, int(time.Since(start).Seconds())))
				}
				u.mu.Unlock()
			}
		}
	}()
	err := f()
	close(stop)
	<-stopped
	u.clear()
	return err
}
