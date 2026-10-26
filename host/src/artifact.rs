use anyhow::{bail, ensure, Result};
use std::{fs, io::{Read, Write}, path::Path};

pub fn read_bounded(path: &str, limit: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= limit, "file exceeds {limit} bytes: {path}");
    Ok(bytes)
}
pub fn paths(out: &str) -> [String; 3] {
    [out.to_owned(), format!("{out}.wrapper.bin"), format!("{out}.journal.bin")]
}
pub fn refuse_existing(out: &str) -> Result<()> {
    for path in paths(out) {
        match fs::symlink_metadata(&path) {
            Ok(_) => bail!("refuse to overwrite artifact: {path}"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
pub fn write(out: &str, bundle: &[u8], wrapper: &[u8], journal: &[u8]) -> Result<()> {
    // create_new is authoritative, including existing symlinks. Reserve all
    // names first. Roll back only files created by this call on I/O failure.
    let paths = paths(out);
    let mut files = Vec::new();
    let result = (|| -> Result<()> {
        for path in &paths {
            if let Some(parent) = Path::new(path).parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            files.push(fs::OpenOptions::new().create_new(true).write(true).open(path)?);
        }
        for (file, bytes) in files.iter_mut().zip([bundle, wrapper, journal]) {
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        Ok(())
    })();
    let created = files.len();
    drop(files);
    if result.is_err() { for path in &paths[..created] { let _ = fs::remove_file(path); } }
    result
}
