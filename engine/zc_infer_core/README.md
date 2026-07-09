# zc_infer_core

Rust skeleton for the Wohper local inference core I/O subsystem.

This crate covers only:

- `MODEL.bin` header and MoE manifest descriptors;
- 2MB alignment validation;
- direct model opening with `O_DIRECT`;
- `io_uring` setup with SQPOLL;
- registered model file;
- fixed 2MB-aligned ping-pong buffers;
- async dense/expert read submission.

It intentionally does not include tokenizer, matmul, AVX kernels, RAG, or the agent control room.

## Modules

```text
src/model_format.rs  on-disk header, manifest, descriptors
src/direct_io.rs     io_uring, fixed files, fixed buffers
src/scheduler.rs     MoE-aware conditional read scheduler
```

## Next

1. Add converter-generated sample `MODEL.bin`.
2. Add a no-model fake block file test.
3. Add AVX2/AVX-512 dequant kernels behind feature flags.
4. Add io_uring no-syscall SQ tail publication path after compatibility mode works.
