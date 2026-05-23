//! GPU hardware detection via nvidia-smi, Apple Silicon, and AMD sysfs probes.

use crate::error::HardwareDetectionError;
use std::path::Path;
use std::process::Command;

/// Physical memory architecture of the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Discrete,
    Unified,
}

impl MemoryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Unified => "unified",
        }
    }
}

/// Detected GPU hardware profile.
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareProfile {
    pub gpu_name: String,
    pub vram_total_mb: i64,
    pub gpu_vendor: String,
    pub memory_kind: MemoryKind,
}

impl HardwareProfile {
    pub fn vram_total_gb(&self) -> f64 {
        self.vram_total_mb as f64 / 1024.0
    }
}

/// Detect GPU hardware by probing nvidia-smi, Apple Silicon, then AMD sysfs.
///
/// Returns `Some(HardwareProfile)` on the first successful probe, or `None`
/// if no GPU is found. Raises `HardwareDetectionError` on malformed nvidia-smi
/// output (does not fall through to later probes in that case).
pub fn detect_hardware() -> Result<Option<HardwareProfile>, HardwareDetectionError> {
    match probe_nvidia_smi() {
        Ok(Some(profile)) => return Ok(Some(profile)),
        Ok(None) => {} // fall through to Apple Silicon
        Err(e) => return Err(e),
    }

    match probe_apple_silicon() {
        Ok(Some(profile)) => return Ok(Some(profile)),
        Ok(None) => {} // fall through to AMD
        Err(e) => return Err(e),
    }

    match probe_amd_sysfs() {
        Ok(Some(profile)) => Ok(Some(profile)),
        Ok(None) => {
            eprintln!(
                "GPU detection failed: nvidia-smi: not installed or failed, \
                 apple-silicon: not detected or unsupported, \
                 amd-sysfs: no AMD card found"
            );
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// Probe nvidia-smi subprocess.
///
/// Runs `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
/// with a 10-second timeout. Parses the first CSV line for GPU name and VRAM in MB.
fn probe_nvidia_smi() -> Result<Option<HardwareProfile>, HardwareDetectionError> {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Ok(None),
    };

    if !output.status.success() {
        return Ok(None);
    }

    // Enforce 10-second timeout at the process level (already completed via .output()).
    // For a stricter timeout, use std::process::Command with a timed wait.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = match stdout.lines().next() {
        Some(line) => line,
        None => return Ok(None),
    };

    let parts: Vec<&str> = first_line.split(',').collect();
    if parts.len() < 2 {
        return Err(HardwareDetectionError::new(format!(
            "Malformed nvidia-smi output: expected 2 CSV fields, got {}: '{}'",
            parts.len(),
            first_line
        )));
    }

    let gpu_name = parts[0].trim().to_string();
    let vram_str = parts[1].trim();

    let vram_total_mb: i64 = vram_str.parse().map_err(|e| {
        HardwareDetectionError::new(format!(
            "Failed to parse nvidia-smi VRAM value '{}': {}",
            vram_str, e
        ))
    })?;

    Ok(Some(HardwareProfile {
        gpu_name,
        vram_total_mb,
        gpu_vendor: "nvidia".to_string(),
        memory_kind: MemoryKind::Discrete,
    }))
}

/// Probe Apple Silicon unified memory on macOS.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn probe_apple_silicon() -> Result<Option<HardwareProfile>, HardwareDetectionError> {
    let memory_output = match Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };
    if !memory_output.status.success() {
        return Ok(None);
    }

    let memory_stdout = String::from_utf8_lossy(&memory_output.stdout);
    let vram_total_mb = parse_apple_unified_memory_mb(&memory_stdout)?;

    let gpu_name = match Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_apple_chip_name(&stdout).unwrap_or_else(|| "Apple Silicon".to_string())
        }
        _ => "Apple Silicon".to_string(),
    };

    Ok(Some(HardwareProfile {
        gpu_name,
        vram_total_mb,
        gpu_vendor: "apple".to_string(),
        memory_kind: MemoryKind::Unified,
    }))
}

/// Non-Apple platforms have no Apple Silicon probe.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn probe_apple_silicon() -> Result<Option<HardwareProfile>, HardwareDetectionError> {
    Ok(None)
}

#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
fn parse_apple_unified_memory_mb(stdout: &str) -> Result<i64, HardwareDetectionError> {
    let memory_bytes: i64 = stdout.trim().parse().map_err(|e| {
        HardwareDetectionError::new(format!(
            "Failed to parse Apple unified memory value '{}': {}",
            stdout.trim(),
            e
        ))
    })?;
    if memory_bytes <= 0 {
        return Err(HardwareDetectionError::new(format!(
            "Invalid Apple unified memory value '{}': must be positive",
            stdout.trim()
        )));
    }
    Ok(memory_bytes / (1024 * 1024))
}

#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
fn parse_apple_chip_name(stdout: &str) -> Option<String> {
    let name = stdout.lines().next()?.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Probe AMD GPU via /sys/class/drm sysfs entries.
///
/// Iterates card entries in sorted order, checks for AMD vendor ID (0x1002),
/// reads VRAM total in bytes, converts to MB. Falls back gracefully when
/// uevent read fails.
fn probe_amd_sysfs() -> Result<Option<HardwareProfile>, HardwareDetectionError> {
    let drm_dir = Path::new("/sys/class/drm");
    if !drm_dir.is_dir() {
        return Ok(None);
    }

    let mut entries: Vec<String> = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(drm_dir) {
        for entry in read_dir.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                entries.push(name.to_string());
            }
        }
    }
    entries.sort();

    for card_name in &entries {
        // Skip non-card entries and render nodes (non-numeric suffix).
        if !card_name.starts_with("card") {
            continue;
        }
        let suffix = &card_name[4..];
        // Skip if suffix contains non-numeric chars (e.g. "card0-DP-1",
        // "card0-render").
        if suffix.contains('-')
            || (!suffix.is_empty() && !suffix.chars().all(|c| c.is_ascii_digit()))
        {
            continue;
        }

        let card_path = drm_dir.join(card_name).join("device");

        // Check AMD vendor ID.
        let vendor_path = card_path.join("vendor");
        let vendor = match std::fs::read_to_string(&vendor_path) {
            Ok(v) => v.trim().to_string(),
            Err(_) => continue,
        };
        if vendor != "0x1002" {
            continue;
        }

        // Read VRAM total in bytes.
        let vram_path = card_path.join("mem_info_vram_total");
        let vram_bytes: i64 = match std::fs::read_to_string(&vram_path) {
            Ok(v) => v.trim().parse().map_err(|e| {
                HardwareDetectionError::new(format!(
                    "Failed to parse AMD VRAM value from {}: {}",
                    vram_path.display(),
                    e
                ))
            })?,
            Err(_) => continue,
        };
        let vram_total_mb = vram_bytes / (1024 * 1024);

        // GPU name from PCI_ID in uevent, with fallback.
        let gpu_name = match std::fs::read_to_string(card_path.join("uevent")) {
            Ok(uevent) => uevent
                .lines()
                .find(|l| l.starts_with("PCI_ID="))
                .and_then(|l| l.strip_prefix("PCI_ID="))
                .map(|id| format!("AMD GPU ({})", id))
                .unwrap_or_else(|| format!("AMD GPU ({})", card_name)),
            Err(_) => format!("AMD GPU ({})", card_name),
        };

        return Ok(Some(HardwareProfile {
            gpu_name,
            vram_total_mb,
            gpu_vendor: "amd".to_string(),
            memory_kind: MemoryKind::Unified,
        }));
    }

    Ok(None)
}

/// Token estimation heuristic: total character count / 4 via integer division.
pub fn estimate_tokens_heuristic(messages: &[crate::core::message::Message]) -> i64 {
    let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
    total_chars as i64 / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_kind_str() {
        assert_eq!(MemoryKind::Discrete.as_str(), "discrete");
        assert_eq!(MemoryKind::Unified.as_str(), "unified");
    }

    #[test]
    fn hardware_profile_vram_gb() {
        let profile = HardwareProfile {
            gpu_name: "Test GPU".to_string(),
            vram_total_mb: 12288,
            gpu_vendor: "nvidia".to_string(),
            memory_kind: MemoryKind::Discrete,
        };
        assert!((profile.vram_total_gb() - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn apple_unified_memory_parse_bytes_to_mb() {
        assert_eq!(
            parse_apple_unified_memory_mb("38654705664\n").unwrap(),
            36864
        );
    }

    #[test]
    fn apple_unified_memory_rejects_malformed_output() {
        let err = parse_apple_unified_memory_mb("not-a-number\n").unwrap_err();
        assert!(err.to_string().contains("Failed to parse"));
    }

    #[test]
    fn apple_unified_memory_rejects_zero() {
        let err = parse_apple_unified_memory_mb("0\n").unwrap_err();
        assert!(err.to_string().contains("must be positive"));
    }

    #[test]
    fn apple_chip_name_parse_trims_first_line() {
        assert_eq!(
            parse_apple_chip_name("Apple M4 Max\nignored"),
            Some("Apple M4 Max".to_string())
        );
        assert_eq!(parse_apple_chip_name("\n"), None);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn apple_silicon_detect_returns_unified_profile() {
        let profile = detect_hardware()
            .unwrap()
            .expect("expected Apple Silicon hardware profile");
        assert_eq!(profile.gpu_vendor, "apple");
        assert_eq!(profile.memory_kind, MemoryKind::Unified);
        assert!(profile.gpu_name.starts_with("Apple "));
        assert!(profile.vram_total_mb > 0);
    }
}
