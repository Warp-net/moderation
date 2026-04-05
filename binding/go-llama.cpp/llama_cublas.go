//go:build cublas
// +build cublas

package go_llama_cpp

/*
#cgo LDFLAGS: -lcublas -lcudart -L/usr/local/cuda/lib64/
*/
import "C"
