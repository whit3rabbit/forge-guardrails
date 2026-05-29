//! Per-model sampling defaults sourced from HuggingFace model cards.
//!
//! `MODEL_SAMPLING_DEFAULTS` is a static map of recommended sampling parameters.
//! Each entry is keyed by the model identity string callers use. All forms
//! point at independent rows so vendor-specific guidance can diverge.
//!
//! Two functions operate on the map:
//! - `get_sampling_defaults`: pure lookup, no side effects.
//! - `apply_sampling_defaults`: policy layer with strict flag, logging, errors.

use std::collections::HashSet;
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::error::UnsupportedModelError;

mod defaults;

#[cfg(test)]
mod tests;

pub use defaults::MODEL_SAMPLING_DEFAULTS;

/// Internal tracking for one-shot INFO log per (model, process) pair.
static INFO_LOGGED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// Pure lookup function for recommended sampling params.
///
/// Returns a fresh `Map` copy of the map entry for known models, or an empty
/// map for unknown models. No logging, no raising, no side effects.
pub fn get_sampling_defaults(model: &str) -> Map<String, Value> {
    match MODEL_SAMPLING_DEFAULTS.get(model) {
        Some(entry) => entry.clone(),
        None => Map::new(),
    }
}

/// Policy-layer function called by client constructors at instantiation time.
///
/// Implements a four-quadrant behavior based on the `strict` flag and model
/// presence in the defaults map:
///
/// | strict | known  | result                           |
/// |--------|--------|----------------------------------|
/// | true   | yes    | fresh copy of the map entry      |
/// | true   | no     | `UnsupportedModelError` raised   |
/// | false  | yes    | empty map, one-shot INFO log     |
/// | false  | no     | empty map, silent                |
pub fn apply_sampling_defaults(
    model: &str,
    strict: bool,
) -> Result<Map<String, Value>, UnsupportedModelError> {
    let defaults = get_sampling_defaults(model);
    let known = !defaults.is_empty();

    if strict {
        if known {
            Ok(defaults)
        } else {
            Err(UnsupportedModelError::new(model))
        }
    } else if known {
        fire_one_shot_info(model);
        Ok(Map::new())
    } else {
        Ok(Map::new())
    }
}

/// Fire a one-shot INFO log per (model, process) pair.
///
/// Uses a process-global Mutex to track which models have already been
/// logged. The log fires once and is silent on subsequent calls for the
/// same model within the same process.
fn fire_one_shot_info(model: &str) {
    // If the mutex is poisoned, skip logging rather than panic.
    let Ok(mut guard) = INFO_LOGGED.lock() else {
        return;
    };
    let logged = guard.get_or_insert_with(HashSet::new);
    if logged.insert(model.to_string()) {
        log::info!(
            "Model '{}' has recommended sampling defaults. \
             Consider opting in with strict mode for optimal behavior.",
            model
        );
    }
}
