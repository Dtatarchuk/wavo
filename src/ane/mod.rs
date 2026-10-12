// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The Neural Engine backend (macOS): a family's encoder written from its GGUF as a Core ML
//! program with one fixed 15 s window, compiled once by `prepare` and run through Core ML. The
//! frontend and the decoder stay on the CPU.
//!
//! Compiled models live in `~/Library/Caches/wavo/`, outside the Hugging Face cache, named by a
//! hash of the GGUF's content and the graph version: `<hash>.v<GRAPH>.mlmodelc`.

pub mod conformer;
#[cfg_attr(not(target_os = "macos"), path = "unsupported.rs")]
mod coreml;
mod mil;
mod proto;

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

pub use conformer::Graph;
pub use coreml::{Model, available};
pub use mil::{Program, Var};

use crate::gguf::Gguf;
use crate::{Error, PrepareStep, Result};

/// The audio window of every program, in samples (15 s): most dictation is shorter, and longer
/// audio is split at pauses into pieces of up to this.
pub const WINDOW: usize = 15 * 16000;
/// Bumped whenever a program changes, so models compiled by an older wavo are rebuilt.
const GRAPH: u32 = 1;

/// Writes a family's encoder into the program and returns its outputs.
pub type Build = fn(&Gguf, &mut Program) -> Result<Vec<Var>>;

fn unsupported(why: &str) -> Error {
  Error::NeuralEngine(why.into())
}

fn cache() -> Result<PathBuf> {
  let home = std::env::var_os("HOME").filter(|h| !h.is_empty());
  let home = home.ok_or_else(|| unsupported("HOME is not set"))?;
  Ok(PathBuf::from(home).join("Library/Caches/wavo"))
}

/// FNV-1a over the file's size and its first and last MiB: the header names every tensor with
/// its offset, and the tail holds weights, so another file of the same name or size differs.
fn hash(path: &Path) -> Result<String> {
  const PART: u64 = 1 << 20;
  let mut f = File::open(path)?;
  let len = f.metadata()?.len();
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  let mut eat = |bytes: &[u8]| {
    for &b in bytes {
      h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
  };
  eat(&len.to_le_bytes());
  let mut buf = vec![0; PART.min(len) as usize];
  f.read_exact(&mut buf)?;
  eat(&buf);
  f.seek(SeekFrom::Start(len - buf.len() as u64))?;
  f.read_exact(&mut buf)?;
  eat(&buf);
  Ok(format!("{h:016x}"))
}

/// The compiled model of a GGUF file, whether it exists or not.
fn compiled(path: &Path) -> Result<PathBuf> {
  Ok(cache()?.join(format!("{}.v{GRAPH}.mlmodelc", hash(path)?)))
}

/// Whether the GGUF's compiled model is in the cache, on a Mac that can run it.
pub fn prepared(path: &Path) -> bool {
  coreml::available() && compiled(path).is_ok_and(|p| p.is_dir())
}

/// Loads the GGUF's compiled model.
pub fn load(path: &Path) -> Result<Model> {
  if !coreml::available() {
    return Err(unsupported(
      "no Neural Engine: it needs an Apple Silicon Mac with macOS 15 or later",
    ));
  }
  let compiled = compiled(path)?;
  if !compiled.is_dir() {
    return Err(unsupported(
      "the model is not prepared for the Neural Engine: run `wavo pull` or wavo::Model::prepare",
    ));
  }
  Model::load(&compiled)
}

/// Builds and compiles the GGUF's program unless it is in the cache, then loads it once so the
/// system compiles and caches it for the Neural Engine. `false` on a Mac without one.
pub fn prepare(
  path: &Path,
  g: &Gguf,
  build: Build,
  mut progress: impl FnMut(PrepareStep) -> ControlFlow<()>,
) -> Result<bool> {
  if !coreml::available() {
    return Ok(false);
  }
  let mut step = |s| if progress(s).is_break() { Err(Error::Cancelled) } else { Ok(()) };
  let target = compiled(path)?;
  let dir = target.parent().unwrap();
  fs::create_dir_all(dir)?;
  let lock = File::create(target.with_extension("lock"))?;
  lock.lock()?;
  if !target.is_dir() {
    step(PrepareStep::Build)?;
    let tmp = target.with_extension(format!("tmp{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let built = (|| {
      let package = tmp.join("model.mlpackage");
      let mut p = Program::new(&package.join("Data/com.apple.CoreML/weights/weight.bin"))?;
      let outputs = build(g, &mut p)?;
      p.finish(&outputs.iter().collect::<Vec<_>>(), &package)?;
      step(PrepareStep::Compile)?;
      let out = coreml::compile(&package)?;
      fs::remove_dir_all(&package)?;
      move_dir(&out, &target)
    })();
    let _ = fs::remove_dir_all(&tmp);
    built?;
  }
  step(PrepareStep::Load)?;
  Model::load(&target)?;
  Ok(true)
}

/// What names the GGUF's compiled models, for `remove`; `None` off macOS, where there are none.
pub fn key(path: &Path) -> Result<Option<String>> {
  if cfg!(target_os = "macos") { hash(path).map(Some) } else { Ok(None) }
}

/// Deletes the compiled models of every graph version named by `key`; returns the bytes freed.
pub fn remove(key: &str) -> Result<u64> {
  let Ok(dir) = cache() else { return Ok(0) };
  let mut freed = 0;
  for e in fs::read_dir(&dir).into_iter().flatten().flatten() {
    if e.file_name().to_string_lossy().starts_with(&format!("{key}.")) {
      freed += size(&e.path());
      if e.path().is_dir() { fs::remove_dir_all(e.path())? } else { fs::remove_file(e.path())? }
    }
  }
  Ok(freed)
}

/// Deletes the GGUF's compiled models; returns the bytes freed.
pub fn unprepare(path: &Path) -> Result<u64> {
  key(path)?.map_or(Ok(0), |key| remove(&key))
}

fn size(path: &Path) -> u64 {
  match fs::read_dir(path) {
    Ok(entries) => entries.flatten().map(|e| size(&e.path())).sum(),
    Err(_) => fs::metadata(path).map_or(0, |m| m.len()),
  }
}

/// Renames a folder, or copies it where renaming fails (another volume).
fn move_dir(from: &Path, to: &Path) -> Result<()> {
  if fs::rename(from, to).is_ok() {
    return Ok(());
  }
  let tmp = to.with_extension("copy");
  copy_dir(from, &tmp)?;
  fs::rename(&tmp, to)?;
  Ok(fs::remove_dir_all(from)?)
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
  fs::create_dir_all(to)?;
  for e in fs::read_dir(from)? {
    let e = e?;
    if e.file_type()?.is_dir() {
      copy_dir(&e.path(), &to.join(e.file_name()))?;
    } else {
      fs::copy(e.path(), to.join(e.file_name()))?;
    }
  }
  Ok(())
}

/// 1 for the first `valid` of `len` frames, 0 for the padding.
pub fn mask(len: usize, valid: usize) -> Vec<f32> {
  (0..len).map(|i| (i < valid) as u32 as f32).collect()
}

/// An additive attention mask over keys: 0 for the first `valid` of `len`, −10⁴ for the padding.
pub fn key_mask(len: usize, valid: usize) -> Vec<f32> {
  (0..len).map(|i| if i < valid { 0.0 } else { -1e4 }).collect()
}
