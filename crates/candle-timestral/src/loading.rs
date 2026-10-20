//! Load exported checkpoints: a directory with `config.json` and `model.safetensors`, on
//! disk or (with the `hub` feature) in a Hugging Face Hub repo.

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Shape, Tensor, safetensors::MmapedSafetensors};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};

use crate::{Config, Error, Result, TimestralModel};

pub const CONFIG_FILE: &str = "config.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// Build the model from `<dir>/config.json` and load `<dir>/model.safetensors` (float32).
///
/// Loading is strict: a missing, misshapen or unused tensor is an error.
pub fn load_model(dir: impl AsRef<Path>, device: &Device) -> Result<TimestralModel> {
    let dir = dir.as_ref();
    let config = Config::from_file(dir.join(CONFIG_FILE))?;
    // SAFETY: the weights file is memory-mapped read-only and must not change while loaded.
    let st = unsafe { MmapedSafetensors::new(dir.join(WEIGHTS_FILE))? };
    let names: Vec<String> = st.tensors().into_iter().map(|(name, _)| name).collect();
    let used = Arc::new(Mutex::new(HashSet::new()));
    let backend = Recording {
        inner: st,
        used: used.clone(),
    };
    let vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device.clone());
    let model = TimestralModel::new(config, vb)?;

    let used = used.lock().unwrap();
    let mut unused: Vec<&String> = names.iter().filter(|n| !used.contains(*n)).collect();
    if !unused.is_empty() {
        unused.sort();
        return Err(Error::Checkpoint(format!(
            "unexpected tensors in {WEIGHTS_FILE}: {unused:?}"
        )));
    }
    Ok(model)
}

/// Download `config.json` and `model.safetensors` of a Hub repo (e.g. `mistralai/Timestral-5M`)
/// to the Hugging Face cache, or find them there; returns their directory.
#[cfg(feature = "hub")]
pub fn download(repo_id: &str) -> Result<PathBuf> {
    let (owner, name) = hf_hub::split_id(repo_id);
    let repo = hf_hub::HFClientSync::new()?.model(owner, name);
    let config = repo.download_file().filename(CONFIG_FILE).send()?;
    repo.download_file().filename(WEIGHTS_FILE).send()?;
    Ok(config.parent().map(Path::to_path_buf).unwrap_or_default())
}

/// Resolve a local checkpoint directory, or a Hub repo id with the `hub` feature.
pub fn resolve_checkpoint(path_or_repo_id: &str) -> Result<PathBuf> {
    let path = PathBuf::from(path_or_repo_id);
    if path.is_dir() {
        return Ok(path);
    }
    if path.exists() {
        return Err(Error::Input(format!(
            "{path_or_repo_id:?} is a file; pass the directory holding {CONFIG_FILE} and {WEIGHTS_FILE}"
        )));
    }
    #[cfg(feature = "hub")]
    return download(path_or_repo_id);
    #[cfg(not(feature = "hub"))]
    Err(Error::Input(format!(
        "{path_or_repo_id:?} is not a local directory (enable the `hub` feature to download from the Hugging Face Hub)"
    )))
}

/// Records the tensors the model reads, to catch checkpoint tensors it never uses.
struct Recording {
    inner: MmapedSafetensors,
    used: Arc<Mutex<HashSet<String>>>,
}

impl SimpleBackend for Recording {
    fn get(
        &self,
        s: Shape,
        name: &str,
        h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.used.lock().unwrap().insert(name.to_string());
        SimpleBackend::get(&self.inner, s, name, h, dtype, dev)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.used.lock().unwrap().insert(name.to_string());
        SimpleBackend::get_unchecked(&self.inner, name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}
