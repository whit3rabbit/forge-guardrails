use std::path::{Path, PathBuf};

use crate::error::BackendError;

pub(super) fn validate_llamafile_runtime_path(path: &Path) -> Result<PathBuf, BackendError> {
    if !path.is_absolute() {
        return Err(BackendError::new(
            0,
            "llamafile_runtime must be an absolute path",
        ));
    }

    let canonical = std::fs::canonicalize(path).map_err(|e| {
        BackendError::new(
            0,
            format!(
                "Cannot resolve llamafile_runtime '{}': {}",
                path.display(),
                e
            ),
        )
    })?;
    let metadata = std::fs::metadata(&canonical).map_err(|e| {
        BackendError::new(
            0,
            format!(
                "Cannot read llamafile_runtime '{}': {}",
                canonical.display(),
                e
            ),
        )
    })?;

    if !metadata.is_file() {
        return Err(BackendError::new(
            0,
            format!(
                "llamafile_runtime must be a regular file: {}",
                canonical.display()
            ),
        ));
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        if metadata.permissions().mode() & 0o111 == 0 {
            return Err(BackendError::new(
                0,
                format!(
                    "llamafile_runtime must be executable: {}",
                    canonical.display()
                ),
            ));
        }
    }

    Ok(canonical)
}
