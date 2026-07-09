use crate::{is_aligned_2mb, ALIGN_2MB};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const MODEL_MAGIC: [u8; 8] = *b"ZCINF01\0";
pub const QUANT_BLOCK_MAGIC: [u8; 8] = *b"ZCBLK01\0";
pub const FORMAT_VERSION: u32 = 1;
pub const QUANT_BLOCK_VERSION: u32 = 1;
pub const QUANT_BLOCK_HEADER_SIZE: usize = 40;
pub const QUANT_TENSOR_RECORD_SIZE: usize = 52;
pub const LAYER_BLOCK_COMPUTE: u32 = 0;
pub const LAYER_BLOCK_GLOBAL_AUX: u32 = 1;
pub const MODEL_FAMILY_FAKE: u32 = 0;
pub const MODEL_FAMILY_GLM52: u32 = 1;
pub const MODEL_FAMILY_DEEPSEEK_V4_FLASH: u32 = 2;

pub fn model_family_name(model_family: u32) -> &'static str {
    match model_family {
        MODEL_FAMILY_FAKE => "fake",
        MODEL_FAMILY_GLM52 => "glm52",
        MODEL_FAMILY_DEEPSEEK_V4_FLASH => "deepseek_v4_flash",
        _ => "unknown",
    }
}

#[derive(Debug, Error)]
pub enum ModelFormatError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic: {0:?}")]
    InvalidMagic([u8; 8]),
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("unaligned field {field}: value={value}, required={ALIGN_2MB}")]
    Unaligned { field: &'static str, value: u64 },
    #[error("manifest is truncated")]
    TruncatedManifest,
    #[error("invalid layer count: header={header}, manifest={manifest}")]
    LayerCountMismatch { header: u32, manifest: u32 },
    #[error("invalid expert table for layer {layer_id}")]
    InvalidExpertTable { layer_id: u32 },
    #[error("shard map error: {0}")]
    ShardMap(String),
    #[error("missing expert shard: layer={layer_id}, expert={expert_id}, path={path}")]
    MissingExpertShard {
        layer_id: u32,
        expert_id: u32,
        path: String,
    },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EngineHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub endian: u32,
    pub file_size: u64,
    pub manifest_offset: u64,
    pub manifest_size: u64,
    pub tokenizer_offset: u64,
    pub tokenizer_size: u64,
    pub router_metadata_offset: u64,
    pub router_metadata_size: u64,
    pub model_family: u32,
    pub architecture: u32,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub experts_per_layer: u32,
    pub active_experts_per_token: u32,
    pub block_alignment: u32,
    pub disk_quant_format: u32,
    pub manifest_checksum: u64,
    pub file_checksum: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ManifestHeader {
    pub layer_count: u32,
    pub expert_count: u32,
    pub tensor_count: u32,
    pub reserved: u32,
    pub layer_desc_offset: u64,
    pub expert_desc_offset: u64,
    pub tensor_desc_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerBlockDescDisk {
    pub layer_id: u32,
    pub flags: u32,
    pub dense_offset: u64,
    pub dense_disk_bytes: u64,
    pub dense_payload_bytes: u64,
    pub dense_dequant_bytes: u64,
    pub tensor_count: u32,
    pub first_tensor_index: u32,
    pub first_expert_index: u32,
    pub num_experts: u32,
    pub quant_format: u32,
    pub checksum_kind: u32,
    pub checksum: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpertBlockDescDisk {
    pub layer_id: u32,
    pub expert_id: u32,
    pub disk_offset: u64,
    pub disk_bytes: u64,
    pub payload_bytes: u64,
    pub dequant_bytes: u64,
    pub quant_format: u32,
    pub route_rank_hint: u32,
    pub checksum: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantBlockHeaderDisk {
    pub magic: [u8; 8],
    pub version: u32,
    pub tensor_count: u32,
    pub quant_format: u32,
    pub flags: u32,
    pub record_table_offset: u64,
    pub names_offset: u64,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantTensorRecordDisk {
    pub dtype_original: u16,
    pub quant_format: u16,
    pub rank: u32,
    pub flags: u32,
    pub name_offset: u64,
    pub shape_offset: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub scale: f32,
    pub zero_point: f32,
}

#[derive(Clone, Debug)]
pub struct MoELayerBlockDesc {
    pub layer_id: u32,
    pub flags: u32,
    pub dense_offset: u64,
    pub dense_disk_bytes: u64,
    pub dense_payload_bytes: u64,
    pub dense_dequant_bytes: u64,
    pub experts: Vec<ExpertBlockDescDisk>,
}

#[derive(Clone, Debug)]
pub struct ExpertShardRef {
    pub layer_id: u32,
    pub expert_id: u32,
    pub path: PathBuf,
    pub disk_bytes: u64,
    pub payload_bytes: u64,
    pub checksum: u64,
}

/// One expert's location inside the consolidated experts pack (T1b: the
/// 11k individual shard files collapse into a single file read through the
/// io_uring ring - no per-expert open/close, fully async).
#[derive(Clone, Copy, Debug)]
pub struct ExpertPackEntry {
    pub offset: u64,
    pub disk_bytes: u64,
    pub payload_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ExpertPackInfo {
    pub file: PathBuf,
    pub entries: HashMap<(u32, u32), ExpertPackEntry>,
}

#[derive(Clone, Debug)]
pub struct ModelManifest {
    pub header: EngineHeader,
    pub layers: Vec<MoELayerBlockDesc>,
    pub base_dir: PathBuf,
    pub expert_shards: HashMap<(u32, u32), ExpertShardRef>,
    pub expert_pack: Option<ExpertPackInfo>,
}

impl ModelManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelFormatError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let header: EngineHeader = read_pod(&mut file, 0)?;
        validate_header(&header)?;

        let manifest_header: ManifestHeader = read_pod(&mut file, header.manifest_offset)?;
        if manifest_header.layer_count != header.num_layers {
            return Err(ModelFormatError::LayerCountMismatch {
                header: header.num_layers,
                manifest: manifest_header.layer_count,
            });
        }

        let layers_disk: Vec<LayerBlockDescDisk> = read_pod_array(
            &mut file,
            header.manifest_offset + manifest_header.layer_desc_offset,
            manifest_header.layer_count as usize,
        )?;
        let experts_disk: Vec<ExpertBlockDescDisk> = read_pod_array(
            &mut file,
            header.manifest_offset + manifest_header.expert_desc_offset,
            manifest_header.expert_count as usize,
        )?;

        let mut layers = Vec::with_capacity(layers_disk.len());
        for layer in layers_disk {
            validate_layer(&layer)?;
            let start = layer.first_expert_index as usize;
            let end = start + layer.num_experts as usize;
            if end > experts_disk.len() {
                return Err(ModelFormatError::InvalidExpertTable {
                    layer_id: layer.layer_id,
                });
            }

            let mut experts = Vec::with_capacity(layer.num_experts as usize);
            for expert in &experts_disk[start..end] {
                validate_expert(expert)?;
                if expert.layer_id != layer.layer_id {
                    return Err(ModelFormatError::InvalidExpertTable {
                        layer_id: layer.layer_id,
                    });
                }
                experts.push(*expert);
            }

            layers.push(MoELayerBlockDesc {
                layer_id: layer.layer_id,
                flags: layer.flags,
                dense_offset: layer.dense_offset,
                dense_disk_bytes: layer.dense_disk_bytes,
                dense_payload_bytes: layer.dense_payload_bytes,
                dense_dequant_bytes: layer.dense_dequant_bytes,
                experts,
            });
        }

        let base_dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
        let expert_shards =
            load_optional_shard_map(path, &base_dir, &mut layers, sidecar_expert_merge_enabled())?;
        let expert_pack = load_optional_expert_pack(path)?;

        Ok(Self {
            header,
            layers,
            base_dir,
            expert_shards,
            expert_pack,
        })
    }

    pub fn expert_shard(&self, layer_id: u32, expert_id: u32) -> Option<&ExpertShardRef> {
        self.expert_shards.get(&(layer_id, expert_id))
    }

    pub fn expert_pack_entry(&self, layer_id: u32, expert_id: u32) -> Option<ExpertPackEntry> {
        self.expert_pack
            .as_ref()
            .and_then(|pack| pack.entries.get(&(layer_id, expert_id)).copied())
    }

    pub fn expert_pack_file(&self) -> Option<&Path> {
        self.expert_pack.as_ref().map(|pack| pack.file.as_path())
    }

    pub fn is_streaming_expert(&self, layer_id: u32, expert_id: u32) -> bool {
        self.expert_shards.contains_key(&(layer_id, expert_id))
    }
}

impl MoELayerBlockDesc {
    pub fn block_type(&self) -> u32 {
        self.flags & 0x0000_00ff
    }

    pub fn is_global_auxiliary(&self) -> bool {
        self.block_type() == LAYER_BLOCK_GLOBAL_AUX
    }

    pub fn is_compute_block(&self) -> bool {
        self.block_type() == LAYER_BLOCK_COMPUTE
    }
}

#[derive(Debug, Deserialize)]
struct ShardMapDisk {
    #[serde(default)]
    format: String,
    #[serde(default)]
    experts: Vec<ExpertShardDisk>,
}

#[derive(Debug, Deserialize)]
struct ExpertShardDisk {
    layer_id: u32,
    expert_id: u32,
    path: String,
    disk_bytes: u64,
    payload_bytes: u64,
    #[serde(default)]
    dequant_bytes: u64,
    #[serde(default)]
    quant_format: u32,
    #[serde(default)]
    checksum: u64,
}

/// When `ZC_SIDECAR_EXPERTS=all` (or `1`), experts present in the sidecar
/// shard map but absent from the compact manifest are merged into the layer
/// expert tables at load time. This lets a catalogseed slice expose its full
/// materialized expert set (for real topK routing) without rewriting the
/// multi-GB `dense_core.bin` manifest.
fn sidecar_expert_merge_enabled() -> bool {
    matches!(
        std::env::var("ZC_SIDECAR_EXPERTS").as_deref(),
        Ok("all") | Ok("1")
    )
}

fn load_optional_shard_map(
    model_path: &Path,
    base_dir: &Path,
    layers: &mut [MoELayerBlockDesc],
    merge_sidecar_experts: bool,
) -> Result<HashMap<(u32, u32), ExpertShardRef>, ModelFormatError> {
    let shard_path = model_path.with_extension("shards.json");
    if !shard_path.exists() {
        return Ok(HashMap::new());
    }

    let text = std::fs::read_to_string(&shard_path)?;
    let disk: ShardMapDisk =
        serde_json::from_str(&text).map_err(|err| ModelFormatError::ShardMap(err.to_string()))?;
    // "zeroclaw-…" is the legacy magic: models converted before the rename
    // keep it on disk and stay valid.
    if disk.format != "wohper-sharded-experts" && disk.format != "zeroclaw-sharded-experts" {
        return Err(ModelFormatError::ShardMap(format!(
            "unsupported shard map format in {}: {}",
            shard_path.display(),
            disk.format
        )));
    }

    let mut known = HashMap::new();
    for layer in layers.iter() {
        for expert in &layer.experts {
            known.insert((expert.layer_id, expert.expert_id), true);
        }
    }

    let mut map = HashMap::new();
    let mut merged = 0usize;
    for expert in disk.experts {
        let in_manifest = known.contains_key(&(expert.layer_id, expert.expert_id));
        if !in_manifest && !merge_sidecar_experts {
            // DeepSeek catalogseed sidecars may include additional converted
            // experts beyond the compact manifest's currently routed set.
            continue;
        }
        validate_aligned("expert_shard.disk_bytes", expert.disk_bytes)?;
        if !in_manifest {
            let Some(layer) = layers
                .iter_mut()
                .find(|layer| layer.is_compute_block() && layer.layer_id == expert.layer_id)
            else {
                continue;
            };
            let template_quant_format = layer
                .experts
                .first()
                .map(|first| first.quant_format)
                .unwrap_or(0);
            layer.experts.push(ExpertBlockDescDisk {
                layer_id: expert.layer_id,
                expert_id: expert.expert_id,
                disk_offset: 0,
                disk_bytes: expert.disk_bytes,
                payload_bytes: expert.payload_bytes,
                dequant_bytes: expert.dequant_bytes,
                quant_format: if expert.quant_format != 0 {
                    expert.quant_format
                } else {
                    template_quant_format
                },
                route_rank_hint: expert.expert_id,
                checksum: expert.checksum,
            });
            merged += 1;
        }
        let path = base_dir.join(&expert.path);
        map.insert(
            (expert.layer_id, expert.expert_id),
            ExpertShardRef {
                layer_id: expert.layer_id,
                expert_id: expert.expert_id,
                path,
                disk_bytes: expert.disk_bytes,
                payload_bytes: expert.payload_bytes,
                checksum: expert.checksum,
            },
        );
    }
    if merged > 0 {
        for layer in layers.iter_mut() {
            layer.experts.sort_by_key(|expert| expert.expert_id);
        }
        eprintln!(
            "sidecar expert merge: {} experts added beyond compact manifest",
            merged
        );
    }
    Ok(map)
}

#[derive(Debug, Deserialize)]
struct ExpertPackDisk {
    #[serde(default)]
    format: String,
    #[serde(default)]
    pack_file: String,
    #[serde(default)]
    entries: Vec<ExpertPackEntryDisk>,
}

#[derive(Debug, Deserialize)]
struct ExpertPackEntryDisk {
    layer_id: u32,
    expert_id: u32,
    offset: u64,
    disk_bytes: u64,
    payload_bytes: u64,
}

/// Loads `<model>.experts_pack.index.json` when present: the consolidated
/// single-file expert pack replaces per-shard file reads.
fn load_optional_expert_pack(model_path: &Path) -> Result<Option<ExpertPackInfo>, ModelFormatError> {
    let index_path = model_path.with_extension("experts_pack.index.json");
    if !index_path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&index_path)?;
    let disk: ExpertPackDisk =
        serde_json::from_str(&text).map_err(|err| ModelFormatError::ShardMap(err.to_string()))?;
    // Legacy magic accepted: packs built before the rename keep it on disk.
    if disk.format != "wohper-experts-pack" && disk.format != "zeroclaw-experts-pack" {
        return Err(ModelFormatError::ShardMap(format!(
            "unsupported experts pack format in {}: {}",
            index_path.display(),
            disk.format
        )));
    }
    let base_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
    let pack_file = base_dir.join(&disk.pack_file);
    if !pack_file.exists() {
        return Err(ModelFormatError::ShardMap(format!(
            "experts pack file missing: {}",
            pack_file.display()
        )));
    }
    let mut entries = HashMap::with_capacity(disk.entries.len());
    for entry in disk.entries {
        validate_aligned("expert_pack.offset", entry.offset)?;
        validate_aligned("expert_pack.disk_bytes", entry.disk_bytes)?;
        entries.insert(
            (entry.layer_id, entry.expert_id),
            ExpertPackEntry {
                offset: entry.offset,
                disk_bytes: entry.disk_bytes,
                payload_bytes: entry.payload_bytes,
            },
        );
    }
    eprintln!(
        "experts_pack loaded: {} experts from {}",
        entries.len(),
        pack_file.display()
    );
    Ok(Some(ExpertPackInfo {
        file: pack_file,
        entries,
    }))
}

fn validate_header(header: &EngineHeader) -> Result<(), ModelFormatError> {
    if header.magic != MODEL_MAGIC {
        return Err(ModelFormatError::InvalidMagic(header.magic));
    }
    if header.version != FORMAT_VERSION {
        return Err(ModelFormatError::UnsupportedVersion(header.version));
    }
    if header.block_alignment as u64 != ALIGN_2MB {
        return Err(ModelFormatError::Unaligned {
            field: "block_alignment",
            value: header.block_alignment as u64,
        });
    }
    validate_aligned("manifest_offset", header.manifest_offset)?;
    validate_aligned("tokenizer_offset", header.tokenizer_offset)?;
    validate_aligned("router_metadata_offset", header.router_metadata_offset)?;
    Ok(())
}

fn validate_layer(layer: &LayerBlockDescDisk) -> Result<(), ModelFormatError> {
    validate_aligned("dense_offset", layer.dense_offset)?;
    validate_aligned("dense_disk_bytes", layer.dense_disk_bytes)?;
    Ok(())
}

fn validate_expert(expert: &ExpertBlockDescDisk) -> Result<(), ModelFormatError> {
    if expert.disk_offset == 0 {
        // External expert shards are resolved through MODEL.shards.json.
        validate_aligned("expert.disk_bytes", expert.disk_bytes)?;
        return Ok(());
    }
    validate_aligned("expert.disk_offset", expert.disk_offset)?;
    validate_aligned("expert.disk_bytes", expert.disk_bytes)?;
    Ok(())
}

fn validate_aligned(field: &'static str, value: u64) -> Result<(), ModelFormatError> {
    if !is_aligned_2mb(value) {
        return Err(ModelFormatError::Unaligned { field, value });
    }
    Ok(())
}

fn read_pod<T: Copy + Default>(file: &mut File, offset: u64) -> Result<T, ModelFormatError> {
    let mut value = T::default();
    let bytes = unsafe {
        std::slice::from_raw_parts_mut((&mut value as *mut T).cast::<u8>(), size_of::<T>())
    };
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(bytes)?;
    Ok(value)
}

fn read_pod_array<T: Copy + Default>(
    file: &mut File,
    offset: u64,
    count: usize,
) -> Result<Vec<T>, ModelFormatError> {
    let byte_len = count
        .checked_mul(size_of::<T>())
        .ok_or(ModelFormatError::TruncatedManifest)?;
    let mut values = vec![T::default(); count];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_len)
    };
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(bytes)?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn shard_map_ignores_catalog_entries_outside_compact_manifest() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zc_shard_map_test_{nonce}"));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("dense_core.bin");
        fs::write(&model, []).unwrap();
        fs::write(
            model.with_extension("shards.json"),
            format!(
                r#"{{
  "format": "wohper-sharded-experts",
  "experts": [
    {{"layer_id": 0, "expert_id": 0, "path": "experts/layer0_expert0.zcblk", "disk_bytes": {align}, "payload_bytes": 16, "checksum": 11}},
    {{"layer_id": 0, "expert_id": 8, "path": "experts/layer0_expert8.zcblk", "disk_bytes": {align}, "payload_bytes": 16, "checksum": 88}}
  ]
}}"#,
                align = ALIGN_2MB
            ),
        )
        .unwrap();
        let mut layers = vec![MoELayerBlockDesc {
            layer_id: 0,
            flags: LAYER_BLOCK_COMPUTE,
            dense_offset: ALIGN_2MB,
            dense_disk_bytes: ALIGN_2MB,
            dense_payload_bytes: 16,
            dense_dequant_bytes: 16,
            experts: vec![ExpertBlockDescDisk {
                layer_id: 0,
                expert_id: 0,
                disk_offset: 0,
                disk_bytes: ALIGN_2MB,
                payload_bytes: 16,
                dequant_bytes: 16,
                quant_format: 2400,
                route_rank_hint: 0,
                checksum: 11,
            }],
        }];

        let map = load_optional_shard_map(&model, &dir, &mut layers, false).unwrap();

        assert_eq!(map.len(), 1);
        assert!(map.contains_key(&(0, 0)));
        assert!(!map.contains_key(&(0, 8)));
        assert_eq!(layers[0].experts.len(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shard_map_merge_extends_compute_layer_expert_table() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("zc_shard_map_merge_test_{nonce}"));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("dense_core.bin");
        fs::write(&model, []).unwrap();
        fs::write(
            model.with_extension("shards.json"),
            format!(
                r#"{{
  "format": "wohper-sharded-experts",
  "experts": [
    {{"layer_id": 0, "expert_id": 0, "path": "experts/layer0_expert0.zcblk", "disk_bytes": {align}, "payload_bytes": 16, "dequant_bytes": 16, "quant_format": 2400, "checksum": 11}},
    {{"layer_id": 0, "expert_id": 8, "path": "experts/layer0_expert8.zcblk", "disk_bytes": {align}, "payload_bytes": 16, "dequant_bytes": 16, "quant_format": 2400, "checksum": 88}},
    {{"layer_id": 1, "expert_id": 3, "path": "experts/layer1_expert3.zcblk", "disk_bytes": {align}, "payload_bytes": 16, "dequant_bytes": 16, "quant_format": 2400, "checksum": 33}}
  ]
}}"#,
                align = ALIGN_2MB
            ),
        )
        .unwrap();
        // Global aux block shares layer_id 0 with the first compute block:
        // the merge must target the compute block only.
        let mut layers = vec![
            MoELayerBlockDesc {
                layer_id: 0,
                flags: LAYER_BLOCK_GLOBAL_AUX,
                dense_offset: ALIGN_2MB,
                dense_disk_bytes: ALIGN_2MB,
                dense_payload_bytes: 16,
                dense_dequant_bytes: 16,
                experts: Vec::new(),
            },
            MoELayerBlockDesc {
                layer_id: 0,
                flags: LAYER_BLOCK_COMPUTE,
                dense_offset: ALIGN_2MB,
                dense_disk_bytes: ALIGN_2MB,
                dense_payload_bytes: 16,
                dense_dequant_bytes: 16,
                experts: vec![ExpertBlockDescDisk {
                    layer_id: 0,
                    expert_id: 0,
                    disk_offset: 0,
                    disk_bytes: ALIGN_2MB,
                    payload_bytes: 16,
                    dequant_bytes: 16,
                    quant_format: 2400,
                    route_rank_hint: 0,
                    checksum: 11,
                }],
            },
        ];

        let map = load_optional_shard_map(&model, &dir, &mut layers, true).unwrap();

        // Expert (1, 3) has no matching compute layer in this manifest and
        // must be dropped: it cannot be routed without a layer block.
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(&(0, 8)));
        assert!(!map.contains_key(&(1, 3)));
        assert!(layers[0].experts.is_empty(), "global aux must stay empty");
        assert_eq!(layers[1].experts.len(), 2);
        assert_eq!(layers[1].experts[0].expert_id, 0);
        assert_eq!(layers[1].experts[1].expert_id, 8);
        assert_eq!(layers[1].experts[1].disk_offset, 0);
        assert_eq!(layers[1].experts[1].quant_format, 2400);
        assert_eq!(layers[1].experts[1].dequant_bytes, 16);
        fs::remove_dir_all(&dir).unwrap();
    }
}
