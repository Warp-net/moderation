//go:build llama
// +build llama

package moderation

// This file makes the llama.cpp static libraries self-contained: it adds the
// in-repo native/lib directory to the linker search path so `go build -tags
// llama` works with no LIBRARY_PATH / C_INCLUDE_PATH / LD_LIBRARY_PATH setup.
//
// ${SRCDIR} is expanded by cgo to the absolute path of this package, so the
// path stays correct regardless of where the repo is checked out. The library
// names themselves (-lbinding, -lllama, ...) are declared by the llama-go
// package; here we only contribute the search directory.
//
// Linkage is static (.a archives), so nothing is required at runtime.

// #cgo LDFLAGS: -L${SRCDIR}/native/lib
import "C"
