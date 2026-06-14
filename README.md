# moderation
Is a small Go package that provides on-device content moderation for Warpnet. It runs a Llama Guard model locally through llama.cpp (via the tcpipuk/llama-go cgo binding) and classifies a piece of text as safe or unsafe, returning the violated hazard category when it isn't.

Internally it loads a GGUF model, runs inference through the model's embedded Llama Guard chat template so the safety taxonomy is applied correctly, and decodes deterministically (temperature 0, fixed seed, a short token budget — the classifier only needs to emit something like unsafe\nS9,S2). The output is parsed into a safe/unsafe verdict, and the S-codes are mapped to a human-readable reason.

It's designed to keep moderation fully self-hosted and dependency-free at runtime: no external moderation API, no network calls — the model file is the only requirement. The llama.cpp backend is gated behind a llama build tag and links the native library under native/lib, so projects can compile without the heavy C dependency when moderation isn't needed.

Written in Go (with cgo), licensed under the AGPL-v3.



