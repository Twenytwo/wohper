# Local Inference Engine Spec

Target: run GLM-5.2-class or 70B+ models as background agentic inference on low-RAM machines with large NVMe storage.

The project goal is not interactive chat speed. It is offline task throughput with strict hardware-preservation constraints:

- model weights are read-only;
- runtime must not write model/KV data to SSD;
- page cache pollution must be avoided;
- swap must be avoided;
- context must stay short through RAG and compression;
- NVMe bandwidth is treated as the primary bottleneck to engineer around.

## Core Runtime Position

Use:

- Linux-only first.
- C++20 or Rust for the engine.
- `O_RDONLY | O_DIRECT | O_NOATIME` for model block reads.
- `io_uring` with `IORING_SETUP_SQPOLL`.
- fixed registered files and fixed registered buffers.
- 2MB-aligned model blocks.
- 2MB-aligned memory buffers.
- `mlock` / `mlockall` for critical buffers when permitted.
- no runtime writes to model/KV disk paths.

Avoid:

- Python in the hot path.
- `mmap` for large weights.
- OS page cache as a model cache.
- unbounded KV growth.
- writing logs to disk during benchmark mode.

## MODEL.bin Format

All offsets and heavyweight blocks are 2MB aligned.

```text
MODEL.bin
├── EngineHeader
├── ManifestHeader
├── LayerBlockDesc[num_layers]
├── ExpertBlockDesc[num_experts_total]
├── TensorDesc[num_tensors_total]
├── TokenizerBlob
├── RouterMetadata
├── LayerDenseBlock[0]
├── ExpertBlock[layer=0, expert=0]
├── ExpertBlock[layer=0, expert=1]
├── ...
└── OutputHeadBlock
```

### EngineHeader

```cpp
struct EngineHeader {
    char magic[8];              // "ZCINF01"
    uint32_t version;           // format version
    uint32_t endian;            // 0=little
    uint64_t file_size;
    uint64_t manifest_offset;
    uint64_t manifest_size;
    uint64_t tokenizer_offset;
    uint64_t tokenizer_size;
    uint64_t router_metadata_offset;
    uint64_t router_metadata_size;
    uint32_t model_family;      // GLM, DeepSeek, Qwen, etc.
    uint32_t architecture;      // dense, MoE
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t num_attention_heads;
    uint32_t num_kv_heads;
    uint32_t experts_per_layer;
    uint32_t active_experts_per_token;
    uint32_t block_alignment;   // 2097152
    uint32_t disk_quant_format; // iq2, q2k, fp8, etc.
    uint64_t manifest_checksum;
    uint64_t file_checksum;
};
```

### LayerBlockDesc

One descriptor per transformer layer for dense parts that always run.

```cpp
struct LayerBlockDesc {
    uint32_t layer_id;
    uint32_t flags;
    uint64_t dense_offset;          // 2MB aligned
    uint64_t dense_disk_bytes;      // 2MB padded
    uint64_t dense_payload_bytes;   // real bytes
    uint64_t dense_dequant_bytes;   // bytes after decode
    uint32_t tensor_count;
    uint32_t first_tensor_index;
    uint32_t quant_format;
    uint32_t checksum_kind;
    uint64_t checksum;
};
```

### ExpertBlockDesc

Experts must be individually addressable and contiguous.

```cpp
struct ExpertBlockDesc {
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t disk_offset;         // 2MB aligned
    uint64_t disk_bytes;          // 2MB padded
    uint64_t payload_bytes;       // real compressed bytes
    uint64_t dequant_bytes;       // target RAM bytes
    uint32_t quant_format;        // q2, q4, fp8, mixed
    uint32_t route_rank_hint;     // optional prior frequency rank
    uint64_t checksum;
};
```

### TensorDesc

```cpp
struct TensorDesc {
    char name[96];
    uint32_t dtype_on_disk;
    uint32_t dtype_runtime;
    uint32_t rank;
    uint32_t flags;
    uint64_t shape[8];
    uint64_t block_relative_offset;
    uint64_t disk_bytes;
    uint64_t runtime_bytes;
    float scale;
    float zero_point;
};
```

## Python Converter Specification

Input:

- HF/safetensors checkpoint or existing GGUF-like intermediate.
- tokenizer files.
- architecture metadata.
- quantization policy.

Output:

- one `MODEL.bin`;
- one `MODEL.index.json` for debugging only;
- optional `checksums.json`.

Rules:

1. Dense always-needed tensors are packed per layer.
2. MoE experts are packed as separate 2MB-aligned contiguous blocks.
3. Blocks are sorted by expected access order:
   - layer dense block;
   - high-frequency experts first;
   - low-frequency experts after.
4. Each block is padded to 2MB.
5. Every offset in descriptors is absolute from file start.
6. The converter emits dequant target size, not just disk size.
7. The converter can emit route priors from calibration prompts.
8. No block should require reading unrelated experts.

Pseudocode:

```python
ALIGN = 2 * 1024 * 1024

def align_up(x, align=ALIGN):
    return (x + align - 1) // align * align

def write_aligned(f, payload: bytes) -> tuple[int, int, int]:
    offset = align_up(f.tell())
    f.write(b"\0" * (offset - f.tell()))
    f.write(payload)
    payload_size = len(payload)
    padded_size = align_up(payload_size)
    f.write(b"\0" * (padded_size - payload_size))
    return offset, padded_size, payload_size

def build_model_bin(checkpoint, tokenizer, route_priors, out_path):
    header = placeholder_header()
    manifest = Manifest()

    with open(out_path, "wb") as f:
        f.write(bytes(header))
        reserve_manifest_area(f)

        tokenizer_blob = pack_tokenizer(tokenizer)
        tokenizer_offset, tokenizer_disk, tokenizer_payload = write_aligned(f, tokenizer_blob)

        router_blob = pack_router_metadata(route_priors)
        router_offset, router_disk, router_payload = write_aligned(f, router_blob)

        for layer_id in range(checkpoint.num_layers):
            dense_payload = pack_dense_layer(checkpoint, layer_id)
            dense_offset, dense_disk, dense_payload_size = write_aligned(f, dense_payload)
            manifest.add_layer(layer_id, dense_offset, dense_disk, dense_payload_size)

            experts = checkpoint.experts(layer_id)
            experts = sort_by_route_prior(experts, route_priors[layer_id])
            for expert in experts:
                compressed = quantize_and_pack_expert(expert, policy="q2_mixed")
                off, disk, payload = write_aligned(f, compressed)
                manifest.add_expert(layer_id, expert.id, off, disk, payload, expert.runtime_bytes)

        output_head = pack_output_head(checkpoint)
        output_offset, output_disk, output_payload = write_aligned(f, output_head)
        manifest.set_output_head(output_offset, output_disk, output_payload)

        final_header = make_header(f, manifest, tokenizer_offset, router_offset)
        write_header_and_manifest(f, final_header, manifest)
```

## io_uring Inference Loop

Use registered file + registered buffers:

- compressed buffer A;
- compressed buffer B;
- runtime/dequant buffer A;
- runtime/dequant buffer B;
- optional expert cache arena.

The SQPOLL ring lets the kernel poll the submission queue, reducing submission syscalls. `io_uring` uses shared rings between user and kernel, and SQ polling allows the kernel thread to pick up SQEs without a submit syscall in steady state.

### C++ Data Structures

```cpp
constexpr size_t ALIGN_2MB = 2ull * 1024 * 1024;
constexpr uint32_t QUEUE_DEPTH = 256;

enum class BlockKind : uint8_t {
    DenseLayer,
    Expert,
    OutputHead
};

struct ReadTicket {
    uint64_t id;
    BlockKind kind;
    uint32_t layer_id;
    uint32_t expert_id;
    uint32_t buffer_index;
    uint64_t file_offset;
    uint32_t disk_bytes;
    uint32_t payload_bytes;
};

struct IoBuffer {
    void* compressed;
    size_t compressed_capacity;
    void* runtime;
    size_t runtime_capacity;
    std::atomic<uint64_t> ticket_id{0};
    std::atomic<bool> io_ready{false};
    std::atomic<bool> decode_ready{false};
};

struct RuntimeState {
    io_uring ring;
    int model_fd;
    uint32_t fixed_file_index;
    IoBuffer buffers[2];
    LockFreeQueue<ReadTicket, 1024> decode_queue;
    std::atomic<uint64_t> next_ticket{1};
};
```

### Setup

```cpp
RuntimeState setup_runtime(const char* model_path) {
    RuntimeState rt{};

    rt.model_fd = open(model_path, O_RDONLY | O_DIRECT | O_NOATIME);
    if (rt.model_fd < 0) die("open model");

    io_uring_params params{};
    params.flags = IORING_SETUP_SQPOLL | IORING_SETUP_CQSIZE;
    params.cq_entries = QUEUE_DEPTH * 2;

    int rc = io_uring_queue_init_params(QUEUE_DEPTH, &rt.ring, &params);
    if (rc < 0) die("io_uring_queue_init_params");

    int fds[1] = { rt.model_fd };
    rc = io_uring_register_files(&rt.ring, fds, 1);
    if (rc < 0) die("io_uring_register_files");
    rt.fixed_file_index = 0;

    iovec iov[4];
    allocate_io_buffer(rt.buffers[0], iov[0], iov[1]);
    allocate_io_buffer(rt.buffers[1], iov[2], iov[3]);

    rc = io_uring_register_buffers(&rt.ring, iov, 4);
    if (rc < 0) die("io_uring_register_buffers");

    return rt;
}
```

### Submit Read

```cpp
uint64_t submit_read(RuntimeState& rt, const BlockDesc& block, uint32_t buffer_index) {
    uint64_t id = rt.next_ticket.fetch_add(1);
    IoBuffer& buf = rt.buffers[buffer_index];

    buf.io_ready.store(false, std::memory_order_relaxed);
    buf.decode_ready.store(false, std::memory_order_relaxed);
    buf.ticket_id.store(id, std::memory_order_release);

    auto* sqe = io_uring_get_sqe(&rt.ring);
    if (!sqe) die("SQ full");

    io_uring_prep_read_fixed(
        sqe,
        rt.fixed_file_index,
        buf.compressed,
        block.disk_bytes,
        block.disk_offset,
        buffer_index * 2
    );
    sqe->flags |= IOSQE_FIXED_FILE;

    ReadTicket* ticket = ticket_pool_alloc();
    *ticket = {
        .id = id,
        .kind = block.kind,
        .layer_id = block.layer_id,
        .expert_id = block.expert_id,
        .buffer_index = buffer_index,
        .file_offset = block.disk_offset,
        .disk_bytes = block.disk_bytes,
        .payload_bytes = block.payload_bytes,
    };
    io_uring_sqe_set_data(sqe, ticket);

    // With SQPOLL, the kernel polling thread observes SQ tail updates.
    io_uring_submit(&rt.ring);
    return id;
}
```

In the final no-syscall steady path, replace `io_uring_submit()` with manual SQ tail publication or use liburing carefully. Keep a compatibility mode with `io_uring_submit()` for debugging and kernels where SQPOLL needs wakeups.

### Completion Pump

```cpp
void completion_pump(RuntimeState& rt) {
    while (running) {
        io_uring_cqe* cqe = nullptr;
        int rc = io_uring_peek_cqe(&rt.ring, &cqe);
        if (rc == -EAGAIN) {
            cpu_relax();
            continue;
        }
        if (rc < 0) die("peek_cqe");

        auto* ticket = static_cast<ReadTicket*>(io_uring_cqe_get_data(cqe));
        if (cqe->res < 0 || static_cast<uint32_t>(cqe->res) != ticket->disk_bytes) {
            mark_io_error(*ticket, cqe->res);
        } else {
            IoBuffer& buf = rt.buffers[ticket->buffer_index];
            buf.io_ready.store(true, std::memory_order_release);
            rt.decode_queue.push(*ticket);
        }

        io_uring_cqe_seen(&rt.ring, cqe);
        ticket_pool_free(ticket);
    }
}
```

## SIMD Dequant Pipeline

Disk blocks are compressed. Compute kernels consume runtime layout.

Pipeline:

```text
NVMe -> compressed buffer -> SIMD dequant -> runtime buffer -> CPU/GPU compute
```

### Decode Worker

```cpp
void decode_worker(RuntimeState& rt) {
    while (running) {
        ReadTicket ticket;
        if (!rt.decode_queue.pop(ticket)) {
            cpu_relax();
            continue;
        }

        IoBuffer& buf = rt.buffers[ticket.buffer_index];
        wait_until(buf.io_ready.load(std::memory_order_acquire));

        switch (get_quant_format(ticket)) {
            case QuantFormat::Q2_MOE:
                dequant_q2_moe_avx512(
                    static_cast<const uint8_t*>(buf.compressed),
                    static_cast<float16*>(buf.runtime),
                    ticket.payload_bytes
                );
                break;
            case QuantFormat::FP8:
                dequant_fp8_avx512(...);
                break;
            default:
                dequant_generic_avx2(...);
                break;
        }

        buf.decode_ready.store(true, std::memory_order_release);
    }
}
```

### AVX-512 Sketch

```cpp
void dequant_q2_moe_avx512(const uint8_t* src, fp16* dst, size_t bytes) {
    // Each byte has four 2-bit weights.
    // Use lookup tables or bit unpacking into int16 lanes.
    // Apply per-block scale and optional zero/offset.
    for (size_t i = 0; i < bytes; i += 64) {
        __m512i packed = _mm512_load_si512((const void*)(src + i));
        // unpack 2-bit lanes:
        // w0 = packed & 0x03
        // w1 = (packed >> 2) & 0x03
        // w2 = (packed >> 4) & 0x03
        // w3 = (packed >> 6) & 0x03
        // convert to fp16/fp32, multiply by scale.
        // store in runtime tensor layout expected by matmul kernel.
    }
}
```

## MoE-Aware Conditional Prefetching

For MoE layers:

1. Dense attention/norm block is prefetched normally.
2. Router logits are computed early.
3. Top-k experts are selected.
4. Only selected experts are submitted to io_uring.
5. Optional predictor prefetches likely experts for next layer using prior route history.

### Routing State

```cpp
struct RoutePrediction {
    uint32_t layer_id;
    uint32_t expert_ids[8];
    float probability[8];
    uint32_t count;
};

struct RouteHistory {
    RingBuffer<RoutePrediction, 256> recent;
    ExpertMarkovTable transition;
};
```

### Layer Loop

```cpp
void run_token(RuntimeState& rt, Model& model, KVCache& kv, Activation& x) {
    uint32_t dense_buf = 0;
    uint32_t expert_buf = 1;

    submit_read(rt, model.layer_dense[0], dense_buf);

    for (uint32_t layer = 0; layer < model.num_layers; layer++) {
        IoBuffer& dense = rt.buffers[dense_buf];
        wait_until(dense.decode_ready.load(std::memory_order_acquire));

        run_attention_and_router(model, layer, dense.runtime, kv, x);

        auto selected = topk_router_experts(x.router_logits, model.active_experts_per_token);

        for (auto expert_id : selected) {
            const auto& expert_block = model.experts[layer][expert_id];
            submit_read(rt, expert_block, expert_buf);

            auto predicted = predict_next_layer_experts(route_history, layer, expert_id);
            schedule_low_priority_prefetch(rt, model, predicted);

            IoBuffer& expert = rt.buffers[expert_buf];
            wait_until(expert.decode_ready.load(std::memory_order_acquire));
            run_expert(model, layer, expert_id, expert.runtime, x);

            expert_buf ^= 1;
        }

        merge_expert_outputs(x);

        uint32_t next_layer = layer + 1;
        if (next_layer < model.num_layers) {
            submit_read(rt, model.layer_dense[next_layer], dense_buf);
        }

        dense_buf ^= 1;
    }
}
```

## Scheduling Policy

Priority tiers:

1. required dense next layer;
2. selected current-layer experts;
3. predicted next-layer experts;
4. hot expert cache refresh.

The scheduler can cancel or ignore low-priority completions if the router prediction was wrong. Wrong prediction cost is bounded because the block is read-only and disposable.

## Build Milestones

1. Parse `MODEL.bin` header and manifest.
2. Direct `O_DIRECT` block read into 2MB-aligned buffer.
3. `io_uring` read with registered file and fixed buffer.
4. SQPOLL compatibility mode.
5. Ping-pong dense layer loop.
6. SIMD unpack microbenchmark.
7. MoE expert block addressing.
8. Router-driven expert prefetch.
9. Obsidian/RAG prompt envelope.
10. Background agent loop integration.
