use crate::ALIGN_2MB;
use io_uring::{opcode, squeue, types, IoUring};
use libc::{c_void, iovec, O_DIRECT, O_NOATIME, O_RDONLY};
use std::fs::File;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use thiserror::Error;

pub const QUEUE_DEPTH: u32 = 256;
pub const FIXED_MODEL_FILE: u32 = 0;
pub const BUFFER_FREE: u8 = 0;
pub const BUFFER_IN_FLIGHT: u8 = 1;
pub const BUFFER_READY: u8 = 2;

#[derive(Debug, Error)]
pub enum DirectIoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("io_uring error: {0}")]
    Ring(std::io::Error),
    #[error("allocation failed")]
    AllocationFailed,
    #[error("buffer capacity exceeded: requested={requested}, capacity={capacity}")]
    BufferTooSmall { requested: u64, capacity: usize },
    #[error("submission queue is full")]
    SubmissionQueueFull,
    #[error("invalid buffer count: {0}")]
    InvalidBufferCount(usize),
    #[error("invalid fixed buffer index: {0}")]
    InvalidBufferIndex(usize),
    #[error("no free fixed buffer")]
    NoFreeBuffer,
    #[error("short external read: requested={requested}, actual={actual}")]
    ShortExternalRead { requested: u64, actual: usize },
}

#[derive(Clone, Copy, Debug)]
pub enum BlockKind {
    Dense,
    Expert,
}

#[derive(Clone, Copy, Debug)]
pub struct ReadTicket {
    pub id: u64,
    pub kind: BlockKind,
    pub layer_id: u32,
    pub expert_id: u32,
    pub fixed_buffer_index: u16,
    pub file_offset: u64,
    pub disk_bytes: u32,
    pub payload_bytes: u32,
}

#[derive(Debug)]
pub struct FixedBuffer {
    pub ptr: NonNull<u8>,
    pub capacity: usize,
    pub state: AtomicU8,
    pub ticket_id: AtomicU64,
    pub io_ready: AtomicBool,
    pub decode_ready: AtomicBool,
}

unsafe impl Send for FixedBuffer {}
unsafe impl Sync for FixedBuffer {}

impl FixedBuffer {
    pub fn allocate(capacity: usize) -> Result<Self, DirectIoError> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { libc::posix_memalign(&mut raw, ALIGN_2MB as usize, capacity) };
        if rc != 0 || raw.is_null() {
            return Err(DirectIoError::AllocationFailed);
        }

        // Best effort. This can fail under normal ulimit settings; the caller can
        // still use the buffer, but benchmark mode should assert mlock success.
        unsafe {
            libc::madvise(raw, capacity, libc::MADV_HUGEPAGE);
            libc::mlock(raw, capacity);
        }

        Ok(Self {
            ptr: NonNull::new(raw.cast::<u8>()).ok_or(DirectIoError::AllocationFailed)?,
            capacity,
            state: AtomicU8::new(BUFFER_FREE),
            ticket_id: AtomicU64::new(0),
            io_ready: AtomicBool::new(false),
            decode_ready: AtomicBool::new(false),
        })
    }

    pub fn as_iovec(&self) -> iovec {
        iovec {
            iov_base: self.ptr.as_ptr().cast::<c_void>(),
            iov_len: self.capacity,
        }
    }
}

impl Drop for FixedBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munlock(self.ptr.as_ptr().cast::<c_void>(), self.capacity);
            libc::free(self.ptr.as_ptr().cast::<c_void>());
        }
    }
}

pub struct DirectIoRuntime {
    pub ring: IoUring,
    pub model_file: File,
    /// Consolidated experts pack (T1b), registered as fixed file #1 so
    /// expert reads go through the same async ring as dense blocks.
    pub pack_file: Option<File>,
    pub io_buffers: Vec<FixedBuffer>,
    pub runtime_buffers: Vec<FixedBuffer>,
    buffer_cursor: AtomicUsize,
    in_flight: AtomicU64,
    next_ticket: AtomicU64,
}

impl DirectIoRuntime {
    pub fn open(
        model_path: impl AsRef<Path>,
        io_buffer_bytes: usize,
        runtime_buffer_bytes: usize,
        io_buffer_count: usize,
    ) -> Result<Self, DirectIoError> {
        Self::open_with_pack(
            model_path,
            None::<&Path>,
            io_buffer_bytes,
            runtime_buffer_bytes,
            io_buffer_count,
        )
    }

    pub fn open_with_pack(
        model_path: impl AsRef<Path>,
        expert_pack_path: Option<impl AsRef<Path>>,
        io_buffer_bytes: usize,
        runtime_buffer_bytes: usize,
        io_buffer_count: usize,
    ) -> Result<Self, DirectIoError> {
        if io_buffer_count == 0 || io_buffer_count > u16::MAX as usize {
            return Err(DirectIoError::InvalidBufferCount(io_buffer_count));
        }

        let model_file = open_direct_readonly(model_path.as_ref())?;
        let pack_file = match expert_pack_path {
            Some(path) => Some(open_direct_readonly(path.as_ref())?),
            None => None,
        };

        let ring = IoUring::builder()
            .setup_sqpoll(1000)
            .setup_cqsize(QUEUE_DEPTH * 2)
            .build(QUEUE_DEPTH)
            .map_err(DirectIoError::Ring)?;

        let mut fds = vec![model_file.as_raw_fd()];
        if let Some(pack) = &pack_file {
            fds.push(pack.as_raw_fd());
        }
        ring.submitter()
            .register_files(&fds)
            .map_err(DirectIoError::Ring)?;

        let mut io_buffers = Vec::with_capacity(io_buffer_count);
        for _ in 0..io_buffer_count {
            io_buffers.push(FixedBuffer::allocate(io_buffer_bytes)?);
        }

        let runtime_buffer_count = if runtime_buffer_bytes == 0 {
            0
        } else {
            io_buffer_count
        };
        let mut runtime_buffers = Vec::with_capacity(runtime_buffer_count);
        for _ in 0..runtime_buffer_count {
            runtime_buffers.push(FixedBuffer::allocate(runtime_buffer_bytes)?);
        }

        let mut iovecs = Vec::with_capacity(io_buffers.len() + runtime_buffers.len());
        iovecs.extend(io_buffers.iter().map(FixedBuffer::as_iovec));
        iovecs.extend(runtime_buffers.iter().map(FixedBuffer::as_iovec));

        // Safety: the registered iovecs point to FixedBuffer allocations owned by
        // DirectIoRuntime. They are not moved or freed until after the ring is
        // dropped, so the kernel never sees dangling buffer pointers.
        unsafe {
            ring.submitter()
                .register_buffers(&iovecs)
                .map_err(DirectIoError::Ring)?;
        }

        Ok(Self {
            ring,
            model_file,
            pack_file,
            io_buffers,
            runtime_buffers,
            buffer_cursor: AtomicUsize::new(0),
            in_flight: AtomicU64::new(0),
            next_ticket: AtomicU64::new(1),
        })
    }

    pub fn has_pack(&self) -> bool {
        self.pack_file.is_some()
    }

    pub fn buffer_count(&self) -> usize {
        self.io_buffers.len()
    }

    pub fn in_flight(&self) -> u64 {
        self.in_flight.load(Ordering::Acquire)
    }

    pub fn available_buffers(&self) -> usize {
        self.io_buffers
            .iter()
            .filter(|buffer| buffer.state.load(Ordering::Acquire) == BUFFER_FREE)
            .count()
    }

    pub fn release_buffer(&self, fixed_buffer_index: u16) -> Result<(), DirectIoError> {
        let buffer = self
            .io_buffers
            .get(fixed_buffer_index as usize)
            .ok_or(DirectIoError::InvalidBufferIndex(fixed_buffer_index as usize))?;
        buffer.io_ready.store(false, Ordering::Release);
        buffer.decode_ready.store(false, Ordering::Release);
        buffer.ticket_id.store(0, Ordering::Release);
        buffer.state.store(BUFFER_FREE, Ordering::Release);
        Ok(())
    }

    fn acquire_free_buffer(&self) -> Option<usize> {
        let len = self.io_buffers.len();
        let start = self.buffer_cursor.fetch_add(1, Ordering::Relaxed) % len;
        for offset in 0..len {
            let index = (start + offset) % len;
            let buffer = &self.io_buffers[index];
            if buffer
                .state
                .compare_exchange(
                    BUFFER_FREE,
                    BUFFER_IN_FLIGHT,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                self.buffer_cursor.store((index + 1) % len, Ordering::Release);
                return Some(index);
            }
        }
        None
    }

    pub fn submit_read(
        &mut self,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        file_offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
    ) -> Result<u64, DirectIoError> {
        let fixed_buffer_index = self
            .acquire_free_buffer()
            .ok_or(DirectIoError::NoFreeBuffer)? as u16;
        let ticket = self.submit_read_on_buffer_deferred(
            kind,
            layer_id,
            expert_id,
            file_offset,
            disk_bytes,
            payload_bytes,
            fixed_buffer_index,
        )?;
        self.flush_submissions()?;
        Ok(ticket)
    }

    pub fn submit_read_deferred(
        &mut self,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        file_offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
    ) -> Result<u64, DirectIoError> {
        self.submit_read_deferred_from(
            FIXED_MODEL_FILE,
            kind,
            layer_id,
            expert_id,
            file_offset,
            disk_bytes,
            payload_bytes,
        )
    }

    /// Deferred ring read from a specific registered file (0 = model core,
    /// 1 = experts pack).
    #[allow(clippy::too_many_arguments)]
    pub fn submit_read_deferred_from(
        &mut self,
        fixed_file: u32,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        file_offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
    ) -> Result<u64, DirectIoError> {
        let fixed_buffer_index = self
            .acquire_free_buffer()
            .ok_or(DirectIoError::NoFreeBuffer)? as u16;
        self.submit_read_on_buffer_deferred_from(
            fixed_file,
            kind,
            layer_id,
            expert_id,
            file_offset,
            disk_bytes,
            payload_bytes,
            fixed_buffer_index,
        )
    }

    pub fn submit_read_on_buffer(
        &mut self,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        file_offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
        fixed_buffer_index: u16,
    ) -> Result<u64, DirectIoError> {
        let ticket = self.submit_read_on_buffer_deferred(
            kind,
            layer_id,
            expert_id,
            file_offset,
            disk_bytes,
            payload_bytes,
            fixed_buffer_index,
        )?;
        self.flush_submissions()?;
        Ok(ticket)
    }

    pub fn submit_read_on_buffer_deferred(
        &mut self,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        file_offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
        fixed_buffer_index: u16,
    ) -> Result<u64, DirectIoError> {
        self.submit_read_on_buffer_deferred_from(
            FIXED_MODEL_FILE,
            kind,
            layer_id,
            expert_id,
            file_offset,
            disk_bytes,
            payload_bytes,
            fixed_buffer_index,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_read_on_buffer_deferred_from(
        &mut self,
        fixed_file: u32,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        file_offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
        fixed_buffer_index: u16,
    ) -> Result<u64, DirectIoError> {
        if fixed_file != FIXED_MODEL_FILE && self.pack_file.is_none() {
            return Err(DirectIoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "expert pack file not registered",
            )));
        }
        let buffer = self
            .io_buffers
            .get(fixed_buffer_index as usize)
            .ok_or(DirectIoError::InvalidBufferIndex(fixed_buffer_index as usize))?;
        match buffer.state.load(Ordering::Acquire) {
            BUFFER_IN_FLIGHT => {}
            BUFFER_FREE => {
                if buffer
                    .state
                    .compare_exchange(
                        BUFFER_FREE,
                        BUFFER_IN_FLIGHT,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    return Err(DirectIoError::NoFreeBuffer);
                }
            }
            _ => return Err(DirectIoError::NoFreeBuffer),
        }
        if disk_bytes as usize > buffer.capacity {
            buffer.state.store(BUFFER_FREE, Ordering::Release);
            return Err(DirectIoError::BufferTooSmall {
                requested: disk_bytes,
                capacity: buffer.capacity,
            });
        }

        let ticket_id = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        buffer.ticket_id.store(ticket_id, Ordering::Release);
        buffer.io_ready.store(false, Ordering::Release);
        buffer.decode_ready.store(false, Ordering::Release);

        let ticket = Box::new(ReadTicket {
            id: ticket_id,
            kind,
            layer_id,
            expert_id,
            fixed_buffer_index,
            file_offset,
            disk_bytes: disk_bytes as u32,
            payload_bytes: payload_bytes as u32,
        });
        let user_data = Box::into_raw(ticket) as u64;

        let read_e = opcode::ReadFixed::new(
            types::Fixed(fixed_file),
            buffer.ptr.as_ptr(),
            disk_bytes as u32,
            fixed_buffer_index,
        )
        .offset(file_offset)
        .build()
        .flags(squeue::Flags::FIXED_FILE)
        .user_data(user_data);

        unsafe {
            if self.ring.submission().push(&read_e).is_err() {
                buffer.state.store(BUFFER_FREE, Ordering::Release);
                let _ = Box::from_raw(user_data as *mut ReadTicket);
                return Err(DirectIoError::SubmissionQueueFull);
            }
        }

        self.in_flight.fetch_add(1, Ordering::AcqRel);

        Ok(ticket_id)
    }

    pub fn flush_submissions(&mut self) -> Result<usize, DirectIoError> {
        // With SQPOLL, this is a single compatibility wakeup for the batch.
        // The kernel polling thread normally observes SQ tail updates directly.
        self.ring.submit().map_err(DirectIoError::Ring)
    }

    pub fn load_external_file_into_buffer(
        &mut self,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        path: &Path,
        disk_bytes: u64,
        payload_bytes: u64,
    ) -> Result<ReadTicket, DirectIoError> {
        self.load_external_file_into_buffer_shared(
            kind,
            layer_id,
            expert_id,
            path,
            disk_bytes,
            payload_bytes,
        )
    }

    /// Shared-reference variant of the external load: only touches atomic
    /// buffer state and the raw buffer memory (never the io_uring ring), so
    /// multiple loads can run on scoped threads concurrently - each acquires
    /// its own fixed buffer.
    pub fn load_external_file_into_buffer_shared(
        &self,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        path: &Path,
        disk_bytes: u64,
        payload_bytes: u64,
    ) -> Result<ReadTicket, DirectIoError> {
        let fixed_buffer_index = self
            .acquire_free_buffer()
            .ok_or(DirectIoError::NoFreeBuffer)? as u16;
        let buffer = self
            .io_buffers
            .get(fixed_buffer_index as usize)
            .ok_or(DirectIoError::InvalidBufferIndex(fixed_buffer_index as usize))?;
        if disk_bytes as usize > buffer.capacity {
            buffer.state.store(BUFFER_FREE, Ordering::Release);
            return Err(DirectIoError::BufferTooSmall {
                requested: disk_bytes,
                capacity: buffer.capacity,
            });
        }

        let ticket_id = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        buffer.ticket_id.store(ticket_id, Ordering::Release);
        buffer.io_ready.store(false, Ordering::Release);
        buffer.decode_ready.store(false, Ordering::Release);

        // O_DIRECT first: under full memory pressure (dense+lmhead caches
        // resident) buffered reads stall in page reclaim and serialize; the
        // fixed buffers are 2MB-aligned so direct reads go at raw device
        // speed. Read length is rounded up to the 4096 block size (EOF makes
        // the final short read legal). Fallback to buffered on open failure.
        let aligned_bytes = ((disk_bytes as usize) + 4095) & !4095usize;
        let (mut file, read_len) = match open_direct_readonly(path) {
            Ok(file) if aligned_bytes <= buffer.capacity => (file, aligned_bytes),
            _ => (File::open(path)?, disk_bytes as usize),
        };
        let target =
            unsafe { std::slice::from_raw_parts_mut(buffer.ptr.as_ptr(), read_len) };
        let mut read_total = 0usize;
        while read_total < disk_bytes as usize {
            let read = file.read(&mut target[read_total..])?;
            if read == 0 {
                break;
            }
            read_total += read;
        }
        if read_total < disk_bytes as usize {
            buffer.state.store(BUFFER_FREE, Ordering::Release);
            return Err(DirectIoError::ShortExternalRead {
                requested: disk_bytes,
                actual: read_total,
            });
        }

        buffer.io_ready.store(true, Ordering::Release);
        buffer.state.store(BUFFER_READY, Ordering::Release);
        Ok(ReadTicket {
            id: ticket_id,
            kind,
            layer_id,
            expert_id,
            fixed_buffer_index,
            file_offset: 0,
            disk_bytes: disk_bytes as u32,
            payload_bytes: payload_bytes as u32,
        })
    }

    /// Loads a batch of ranges from the experts PACK into fixed buffers
    /// CONCURRENTLY via positional O_DIRECT preads on the single shared fd.
    /// (The io_uring path for the second registered file stalls on the
    /// WSL2 kernel - completions never arrive; tracked separately. Parallel
    /// preads on one fd keep all of the pack's win: no per-expert open,
    /// no page-cache churn, aligned direct reads.)
    pub fn load_pack_ranges_into_buffers(
        &mut self,
        requests: &[(BlockKind, u32, u32, u64, u64, u64)],
    ) -> Vec<Result<ReadTicket, DirectIoError>> {
        let Some(pack) = &self.pack_file else {
            return requests
                .iter()
                .map(|_| {
                    Err(DirectIoError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "expert pack file not open",
                    )))
                })
                .collect();
        };
        let shared: &Self = self;
        let pack: &File = pack;
        let mut results: Vec<Option<Result<ReadTicket, DirectIoError>>> =
            (0..requests.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (slot, request) in results.iter_mut().zip(requests.iter()) {
                let (kind, layer_id, expert_id, offset, disk_bytes, payload_bytes) = *request;
                scope.spawn(move || {
                    *slot = Some(shared.load_pack_range_shared(
                        pack,
                        kind,
                        layer_id,
                        expert_id,
                        offset,
                        disk_bytes,
                        payload_bytes,
                    ));
                });
            }
        });
        results
            .into_iter()
            .map(|slot| slot.expect("scoped thread filled its slot"))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn load_pack_range_shared(
        &self,
        pack: &File,
        kind: BlockKind,
        layer_id: u32,
        expert_id: u32,
        offset: u64,
        disk_bytes: u64,
        payload_bytes: u64,
    ) -> Result<ReadTicket, DirectIoError> {
        use std::os::unix::fs::FileExt;
        let fixed_buffer_index = self
            .acquire_free_buffer()
            .ok_or(DirectIoError::NoFreeBuffer)? as u16;
        let buffer = &self.io_buffers[fixed_buffer_index as usize];
        if disk_bytes as usize > buffer.capacity {
            buffer.state.store(BUFFER_FREE, Ordering::Release);
            return Err(DirectIoError::BufferTooSmall {
                requested: disk_bytes,
                capacity: buffer.capacity,
            });
        }
        let ticket_id = self.next_ticket.fetch_add(1, Ordering::Relaxed);
        buffer.ticket_id.store(ticket_id, Ordering::Release);
        buffer.io_ready.store(false, Ordering::Release);
        buffer.decode_ready.store(false, Ordering::Release);

        let target =
            unsafe { std::slice::from_raw_parts_mut(buffer.ptr.as_ptr(), disk_bytes as usize) };
        let mut read_total = 0usize;
        while read_total < disk_bytes as usize {
            let read = pack
                .read_at(&mut target[read_total..], offset + read_total as u64)
                .map_err(DirectIoError::Io)?;
            if read == 0 {
                break;
            }
            read_total += read;
        }
        if read_total < disk_bytes as usize {
            buffer.state.store(BUFFER_FREE, Ordering::Release);
            return Err(DirectIoError::ShortExternalRead {
                requested: disk_bytes,
                actual: read_total,
            });
        }
        buffer.io_ready.store(true, Ordering::Release);
        buffer.state.store(BUFFER_READY, Ordering::Release);
        Ok(ReadTicket {
            id: ticket_id,
            kind,
            layer_id,
            expert_id,
            fixed_buffer_index,
            file_offset: offset,
            disk_bytes: disk_bytes as u32,
            payload_bytes: payload_bytes as u32,
        })
    }

    /// Loads a batch of external files into fixed buffers CONCURRENTLY on
    /// scoped threads (one per request, bounded by free buffers upstream).
    /// The sidecar expert reads were the dominant decode cost when issued
    /// sequentially: each is a cold ~14MB buffered read under memory
    /// pressure. Returns per-request results in input order.
    pub fn load_external_files_into_buffers(
        &mut self,
        requests: &[(BlockKind, u32, u32, std::path::PathBuf, u64, u64)],
    ) -> Vec<Result<ReadTicket, DirectIoError>> {
        if requests.len() <= 1 {
            return requests
                .iter()
                .map(|(kind, layer_id, expert_id, path, disk_bytes, payload_bytes)| {
                    self.load_external_file_into_buffer_shared(
                        *kind,
                        *layer_id,
                        *expert_id,
                        path,
                        *disk_bytes,
                        *payload_bytes,
                    )
                })
                .collect();
        }
        let shared: &Self = self;
        let mut results: Vec<Option<Result<ReadTicket, DirectIoError>>> =
            (0..requests.len()).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (slot, request) in results.iter_mut().zip(requests.iter()) {
                let (kind, layer_id, expert_id, path, disk_bytes, payload_bytes) = request;
                scope.spawn(move || {
                    *slot = Some(shared.load_external_file_into_buffer_shared(
                        *kind,
                        *layer_id,
                        *expert_id,
                        path,
                        *disk_bytes,
                        *payload_bytes,
                    ));
                });
            }
        });
        results
            .into_iter()
            .map(|slot| slot.expect("scoped thread filled its slot"))
            .collect()
    }

    pub fn poll_completions<F>(&mut self, mut on_complete: F) -> Result<usize, DirectIoError>
    where
        F: FnMut(ReadTicket, i32),
    {
        let mut count = 0;
        let mut cq = self.ring.completion();
        while let Some(cqe) = cq.next() {
            count += 1;
            let ticket_ptr = cqe.user_data() as *mut ReadTicket;
            let ticket = unsafe { *Box::from_raw(ticket_ptr) };
            let res = cqe.result();

            if res >= 0 && res as u32 == ticket.disk_bytes {
                let buffer = &self.io_buffers[ticket.fixed_buffer_index as usize];
                if buffer.ticket_id.load(Ordering::Acquire) == ticket.id {
                    buffer.io_ready.store(true, Ordering::Release);
                    buffer.state.store(BUFFER_READY, Ordering::Release);
                }
            }
            self.in_flight.fetch_sub(1, Ordering::AcqRel);

            on_complete(ticket, res);
        }
        Ok(count)
    }
}

fn open_direct_readonly(path: &Path) -> Result<File, DirectIoError> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        DirectIoError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path contains NUL",
        ))
    })?;
    let fd: RawFd = unsafe { libc::open(c_path.as_ptr(), O_RDONLY | O_DIRECT | O_NOATIME) };
    if fd < 0 {
        return Err(DirectIoError::Io(std::io::Error::last_os_error()));
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
