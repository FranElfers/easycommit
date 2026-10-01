// Command easycommit proposes Conventional Commits messages for the changes
// of a Git working tree, using a local model served by llama.cpp.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/franelfers/easycommit/internal/cli"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := cli.Run(ctx, os.Args[1:], os.Stdout, os.Stderr)
	stop()
	os.Exit(code)
}
