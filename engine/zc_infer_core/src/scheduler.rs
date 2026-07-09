use crate::compute::{ExpertCacheConfig, ExpertCacheError, ExpertLruCache};
use crate::direct_io::{BlockKind, DirectIoError, DirectIoRuntime, ReadTicket};
use crate::model_format::{ModelManifest, MoELayerBlockDesc};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("direct I/O error: {0}")]
    DirectIo(#[from] DirectIoError),
    #[error("invalid expert index: layer={layer_id}, expert={expert_id}")]
    InvalidExpert { layer_id: u32, expert_id: u32 },
    #[error("expert cache error: {0}")]
    ExpertCache(#[from] ExpertCacheError),
}

#[derive(Clone, Copy, Debug)]
pub struct ExpertRoute {
    pub expert_id: u32,
    pub score: f32,
}

#[derive(Debug)]
pub struct ReadyBlock {
    pub ticket: ReadTicket,
    pub result: i32,
}

#[derive(Clone, Debug)]
pub struct ReadRequest {
    pub kind: BlockKind,
    pub layer_id: u32,
    pub expert_id: u32,
    pub file_offset: u64,
    pub disk_bytes: u64,
    pub payload_bytes: u64,
    pub external_path: Option<PathBuf>,
    /// Registered ring file: 0 = model core, 1 = experts pack (T1b).
    pub fixed_file: u32,
}

pub struct MoEIoScheduler {
    pub io: DirectIoRuntime,
    pub manifest: ModelManifest,
    pending: VecDeque<ReadRequest>,
    ready: VecDeque<ReadyBlock>,
    submitted_reads: AtomicU64,
    expert_cache: Option<ExpertLruCache>,
}

impl MoEIoScheduler {
    pub fn new(io: DirectIoRuntime, manifest: ModelManifest) -> Self {
        Self {
            io,
            manifest,
            pending: VecDeque::with_capacity(1024),
            ready: VecDeque::with_capacity(256),
            submitted_reads: AtomicU64::new(0),
            expert_cache: None,
        }
    }

    pub fn with_expert_cache(mut self, config: ExpertCacheConfig) -> Result<Self, SchedulerError> {
        self.expert_cache = Some(ExpertLruCache::new(config)?);
        Ok(self)
    }

    pub fn submit_dense_read(&mut self, layer_id: u32) -> Result<u64, SchedulerError> {
        let layer = &self.manifest.layers[layer_id as usize];

        let ticket = self.io.submit_read(
            BlockKind::Dense,
            layer.layer_id,
            u32::MAX,
            layer.dense_offset,
            layer.dense_disk_bytes,
            layer.dense_payload_bytes,
        )?;
        self.submitted_reads.fetch_add(1, Ordering::Relaxed);
        Ok(ticket)
    }

    pub fn submit_expert_read(
        &mut self,
        layer_id: u32,
        expert_id: u32,
    ) -> Result<u64, SchedulerError> {
        let layer = &self.manifest.layers[layer_id as usize];
        let expert = layer
            .experts
            .iter()
            .find(|candidate| candidate.expert_id == expert_id)
            .ok_or(SchedulerError::InvalidExpert {
                layer_id,
                expert_id,
            })?;

        let ticket = self.io.submit_read(
            BlockKind::Expert,
            layer.layer_id,
            expert_id,
            expert.disk_offset,
            expert.disk_bytes,
            expert.payload_bytes,
        )?;
        self.submitted_reads.fetch_add(1, Ordering::Relaxed);
        Ok(ticket)
    }

    pub fn submit_selected_experts(
        &mut self,
        layer_id: u32,
        routes: &[ExpertRoute],
    ) -> Result<Vec<u64>, SchedulerError> {
        let mut tickets = Vec::with_capacity(routes.len());
        for route in routes {
            tickets.push(self.submit_expert_read(layer_id, route.expert_id)?);
        }
        Ok(tickets)
    }

    pub fn prefetch_layer_dense_and_predicted_experts(
        &mut self,
        layer_id: u32,
        predicted: &[ExpertRoute],
    ) -> Result<(), SchedulerError> {
        self.submit_dense_read(layer_id)?;
        self.submit_selected_experts(layer_id, predicted)?;
        Ok(())
    }

    pub fn enqueue_dense_read(&mut self, layer_id: u32) {
        let layer = &self.manifest.layers[layer_id as usize];
        self.pending.push_back(ReadRequest {
            kind: BlockKind::Dense,
            layer_id: layer.layer_id,
            expert_id: u32::MAX,
            file_offset: layer.dense_offset,
            disk_bytes: layer.dense_disk_bytes,
            payload_bytes: layer.dense_payload_bytes,
            external_path: None,
            fixed_file: 0,
        });
    }

    pub fn enqueue_expert_read(
        &mut self,
        layer_id: u32,
        expert_id: u32,
    ) -> Result<(), SchedulerError> {
        let layer = &self.manifest.layers[layer_id as usize];
        let expert = layer
            .experts
            .iter()
            .find(|candidate| candidate.expert_id == expert_id)
            .ok_or(SchedulerError::InvalidExpert {
                layer_id,
                expert_id,
            })?;

        // T1b: experts present in the consolidated pack go through the
        // async io_uring ring (fixed file #1) - no per-shard open/close,
        // no synchronous read on the compute thread.
        if let Some(entry) = self.manifest.expert_pack_entry(layer.layer_id, expert_id) {
            self.pending.push_back(ReadRequest {
                kind: BlockKind::Expert,
                layer_id: layer.layer_id,
                expert_id,
                file_offset: entry.offset,
                disk_bytes: entry.disk_bytes,
                payload_bytes: entry.payload_bytes,
                external_path: None,
                fixed_file: 1,
            });
            return Ok(());
        }

        self.pending.push_back(ReadRequest {
            kind: BlockKind::Expert,
            layer_id: layer.layer_id,
            expert_id,
            file_offset: expert.disk_offset,
            disk_bytes: expert.disk_bytes,
            payload_bytes: expert.payload_bytes,
            external_path: self
                .manifest
                .expert_shard(layer.layer_id, expert_id)
                .map(|shard| shard.path.clone()),
            fixed_file: 0,
        });
        Ok(())
    }

    pub fn enqueue_selected_experts(
        &mut self,
        layer_id: u32,
        routes: &[ExpertRoute],
    ) -> Result<(), SchedulerError> {
        for route in routes {
            self.enqueue_expert_read(layer_id, route.expert_id)?;
        }
        Ok(())
    }

    pub fn enqueue_layer_io_plan(
        &mut self,
        layer_id: u32,
        routes: &[ExpertRoute],
    ) -> Result<u64, SchedulerError> {
        let layer = &self.manifest.layers[layer_id as usize];
        let mut bytes = layer.dense_disk_bytes;
        let mut requests = Vec::with_capacity(routes.len() + 1);
        requests.push(ReadRequest {
            kind: BlockKind::Dense,
            layer_id: layer.layer_id,
            expert_id: u32::MAX,
            file_offset: layer.dense_offset,
            disk_bytes: layer.dense_disk_bytes,
            payload_bytes: layer.dense_payload_bytes,
            external_path: None,
            fixed_file: 0,
        });

        for route in routes {
            let expert = layer
                .experts
                .iter()
                .find(|candidate| candidate.expert_id == route.expert_id)
                .ok_or(SchedulerError::InvalidExpert {
                    layer_id,
                    expert_id: route.expert_id,
                })?;
            bytes += expert.disk_bytes;
            requests.push(ReadRequest {
                kind: BlockKind::Expert,
                layer_id: layer.layer_id,
                expert_id: route.expert_id,
                file_offset: expert.disk_offset,
                disk_bytes: expert.disk_bytes,
                payload_bytes: expert.payload_bytes,
                external_path: self
                    .manifest
                    .expert_shard(layer.layer_id, route.expert_id)
                    .map(|shard| shard.path.clone()),
                fixed_file: 0,
            });
        }

        self.pending.extend(requests);
        Ok(bytes)
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn in_flight(&self) -> u64 {
        self.io.in_flight()
    }

    pub fn available_buffers(&self) -> usize {
        self.io.available_buffers()
    }

    pub fn submit_pending_until_full(
        &mut self,
        max_new_submissions: usize,
    ) -> Result<usize, SchedulerError> {
        let mut submitted = 0usize;
        // External-path requests (sidecar shards / disk cache) are drained
        // into a batch and loaded CONCURRENTLY below: issued one-by-one they
        // are cold ~14MB buffered reads on the calling thread and dominated
        // the decode wall clock (V3 profiling).
        let mut external: Vec<ReadRequest> = Vec::new();
        let mut pack_batch: Vec<ReadRequest> = Vec::new();
        while submitted + external.len() + pack_batch.len() < max_new_submissions {
            let Some(request) = self.pending.pop_front() else {
                break;
            };

            if request.external_path.is_some() {
                external.push(request);
                continue;
            }
            if request.fixed_file == 1 {
                // Experts-pack ranges: parallel O_DIRECT preads on the
                // single pack fd (ring completions for a second registered
                // file stall on the WSL2 kernel).
                pack_batch.push(request);
                continue;
            }

            match self.io.submit_read_deferred_from(
                request.fixed_file,
                request.kind,
                request.layer_id,
                request.expert_id,
                request.file_offset,
                request.disk_bytes,
                request.payload_bytes,
            ) {
                Ok(_) => {
                    submitted += 1;
                    self.submitted_reads.fetch_add(1, Ordering::Relaxed);
                }
                Err(DirectIoError::NoFreeBuffer) => {
                    self.pending.push_front(request);
                    break;
                }
                Err(DirectIoError::SubmissionQueueFull) => {
                    self.pending.push_front(request);
                    break;
                }
                Err(err) => return Err(SchedulerError::DirectIo(err)),
            }
        }
        if submitted > 0 {
            self.io.flush_submissions()?;
        }

        if !pack_batch.is_empty() {
            let mut queue: VecDeque<ReadRequest> = pack_batch.into_iter().collect();
            loop {
                let free = self.io.available_buffers();
                if free == 0 || queue.is_empty() {
                    break;
                }
                let take = free.min(queue.len());
                let chunk: Vec<ReadRequest> = queue.drain(..take).collect();
                let specs: Vec<(BlockKind, u32, u32, u64, u64, u64)> = chunk
                    .iter()
                    .map(|request| {
                        (
                            request.kind,
                            request.layer_id,
                            request.expert_id,
                            request.file_offset,
                            request.disk_bytes,
                            request.payload_bytes,
                        )
                    })
                    .collect();
                let results = self.io.load_pack_ranges_into_buffers(&specs);
                let mut out_of_buffers = false;
                for (request, result) in chunk.into_iter().zip(results) {
                    match result {
                        Ok(ticket) => {
                            self.ready
                                .push_back(ReadyBlock { ticket, result: request.disk_bytes as i32 });
                            submitted += 1;
                            self.submitted_reads.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(DirectIoError::NoFreeBuffer) => {
                            self.pending.push_front(request);
                            out_of_buffers = true;
                        }
                        Err(err) => return Err(SchedulerError::DirectIo(err)),
                    }
                }
                if out_of_buffers {
                    break;
                }
            }
            while let Some(request) = queue.pop_back() {
                self.pending.push_front(request);
            }
        }

        if !external.is_empty() {
            // Resolve cache paths first (pass-through when no cache is
            // attached: the shard is read directly).
            let mut queue: VecDeque<(ReadRequest, PathBuf)> =
                VecDeque::with_capacity(external.len());
            for request in external {
                let path = request.external_path.clone().expect("external request");
                let cache_path = if let Some(cache) = &mut self.expert_cache {
                    let source_path = if path.exists() { Some(path.as_path()) } else { None };
                    cache.ensure_expert(
                        request.layer_id,
                        request.expert_id,
                        source_path,
                        Some(request.disk_bytes),
                    )?
                } else {
                    path
                };
                queue.push_back((request, cache_path));
            }
            loop {
                let free = self.io.available_buffers();
                if free == 0 || queue.is_empty() {
                    break;
                }
                let take = free.min(queue.len());
                let chunk: Vec<(ReadRequest, PathBuf)> = queue.drain(..take).collect();
                let specs: Vec<(BlockKind, u32, u32, PathBuf, u64, u64)> = chunk
                    .iter()
                    .map(|(request, path)| {
                        (
                            request.kind,
                            request.layer_id,
                            request.expert_id,
                            path.clone(),
                            request.disk_bytes,
                            request.payload_bytes,
                        )
                    })
                    .collect();
                let results = self.io.load_external_files_into_buffers(&specs);
                let mut out_of_buffers = false;
                for ((request, _path), result) in chunk.into_iter().zip(results) {
                    match result {
                        Ok(ticket) => {
                            self.ready
                                .push_back(ReadyBlock { ticket, result: request.disk_bytes as i32 });
                            submitted += 1;
                            self.submitted_reads.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(DirectIoError::NoFreeBuffer) => {
                            self.pending.push_front(request);
                            out_of_buffers = true;
                        }
                        Err(err) => return Err(SchedulerError::DirectIo(err)),
                    }
                }
                if out_of_buffers {
                    break;
                }
            }
            // Anything left waits for buffers to free up (caller re-submits).
            while let Some((request, _path)) = queue.pop_back() {
                self.pending.push_front(request);
            }
        }
        Ok(submitted)
    }

    pub fn pump_completions(&mut self) -> Result<usize, SchedulerError> {
        let ready = &mut self.ready;
        let count = self.io.poll_completions(|ticket, result| {
            ready.push_back(ReadyBlock { ticket, result });
        })?;
        Ok(count)
    }

    pub fn pop_ready(&mut self) -> Option<ReadyBlock> {
        self.ready.pop_front()
    }

    pub fn release_ready_block(&mut self, block: &ReadyBlock) -> Result<(), SchedulerError> {
        self.io.release_buffer(block.ticket.fixed_buffer_index)?;
        Ok(())
    }

    pub fn run_layer_io_plan(
        &mut self,
        layer_id: u32,
        selected_experts: &[ExpertRoute],
        predicted_next_layer: &[ExpertRoute],
    ) -> Result<(), SchedulerError> {
        self.submit_dense_read(layer_id)?;
        self.submit_selected_experts(layer_id, selected_experts)?;

        let next_layer = layer_id + 1;
        if !predicted_next_layer.is_empty() && (next_layer as usize) < self.manifest.layers.len() {
            self.prefetch_layer_dense_and_predicted_experts(next_layer, predicted_next_layer)?;
        }

        Ok(())
    }

    pub fn layer(&self, layer_id: u32) -> &MoELayerBlockDesc {
        &self.manifest.layers[layer_id as usize]
    }
}
