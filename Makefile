VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X github.com/franelfers/easycommit/internal/cli.Version=$(VERSION)

.PHONY: build check test clean

# Static binary: runs on both glibc and musl.
build:
	CGO_ENABLED=0 go build -trimpath -ldflags "$(LDFLAGS)" -o easycommit .

check:
	@test -z "$$(gofmt -l .)" || { echo "gofmt needed:"; gofmt -l .; exit 1; }
	go vet ./...

test:
	go test -race ./...

clean:
	rm -rf easycommit dist
