use std::collections::{BinaryHeap, HashMap, HashSet};

use super::{
    dictionary_has_meaningful_savings, is_dictionary_compressed_output, DICTIONARY_MAX_DICT_SIZE,
    DICTIONARY_MAX_INPUT_BYTES, DICTIONARY_MIN_ENTRY_SAVINGS_BYTES, DICTIONARY_MIN_OCCURRENCES,
    LZW_DICTIONARY_HEADER as DICTIONARY_HEADER,
};

const MIN_SUBSTRING_LEN: usize = 15;
const MAX_SUBSTRING_LEN: usize = 120;
const MAX_REPETITION_SAMPLES: usize = 500;
const MIN_REPETITION_RATIO_PERCENT: usize = 5;
const ROLLING_HASH_BASE: u64 = 257;
const SELECTED_SUBSTRING_LENGTHS: &[usize] = &[15, 24, 32, 48, 64, 96, 120];

#[derive(Debug, Clone)]
struct Candidate<'a> {
    original: &'a str,
    count: usize,
    savings: isize,
}

#[derive(Debug, Clone)]
struct SelectedEntry<'a> {
    marker: String,
    original: &'a str,
    positions: Vec<usize>,
}

/// Compress repeated one-line substrings into a model-readable dictionary.
pub(super) fn compress_lzw_dictionary(output: &str) -> Option<String> {
    if output.len() > DICTIONARY_MAX_INPUT_BYTES || is_dictionary_compressed_output(output) {
        return None;
    }

    let offsets = char_offsets(output);
    let char_count = offsets.len().saturating_sub(1);
    if char_count < MIN_SUBSTRING_LEN || !has_sufficient_repetition(output, &offsets) {
        return None;
    }

    let nonce = collision_free_nonce(output)?;
    let candidates = find_repeated_substrings(output, &nonce);
    if candidates.is_empty() {
        return None;
    }

    let selected = select_non_overlapping(output, &candidates, &nonce);
    if selected.is_empty() {
        return None;
    }

    let compressed = apply_replacements(output, &selected);
    let dictionary = build_dictionary(&selected)?;
    let result = format!("{DICTIONARY_HEADER}\n{dictionary}\n{compressed}");
    let savings = output.len().checked_sub(result.len())?;
    if !dictionary_has_meaningful_savings(output.len(), savings) {
        return None;
    }
    Some(result)
}

fn char_offsets(value: &str) -> Vec<usize> {
    let mut offsets = value.char_indices().map(|(idx, _)| idx).collect::<Vec<_>>();
    offsets.push(value.len());
    offsets
}

fn has_sufficient_repetition<'a>(text: &'a str, offsets: &[usize]) -> bool {
    let char_count = offsets.len().saturating_sub(1);
    let max_pos = char_count.saturating_sub(MIN_SUBSTRING_LEN);
    if max_pos == 0 {
        return false;
    }

    let step = std::cmp::max(1, max_pos / MAX_REPETITION_SAMPLES);
    let mut seen: HashSet<&'a str> = HashSet::new();
    let mut repeated = 0usize;
    let mut total = 0usize;

    for start in (0..=max_pos).step_by(step) {
        let Some(substring) = slice_chars(text, offsets, start, MIN_SUBSTRING_LEN) else {
            continue;
        };
        if !eligible_substring(substring) {
            continue;
        }
        total += 1;
        if !seen.insert(substring) {
            repeated += 1;
        }
    }

    total > 0 && repeated.saturating_mul(100) / total >= MIN_REPETITION_RATIO_PERCENT
}

fn find_repeated_substrings<'a>(text: &'a str, nonce: &str) -> Vec<Candidate<'a>> {
    let mut counts: HashMap<&'a str, usize> = HashMap::new();

    for len in SELECTED_SUBSTRING_LENGTHS {
        if *len < MIN_SUBSTRING_LEN || *len > MAX_SUBSTRING_LEN || *len > text.len() {
            continue;
        }

        let mut hashed_positions: HashMap<u64, Vec<usize>> = HashMap::new();
        collect_rolling_hash_positions(text.as_bytes(), *len, &mut hashed_positions);

        for positions in hashed_positions.into_values() {
            if positions.len() < DICTIONARY_MIN_OCCURRENCES {
                continue;
            }

            let mut exact: HashMap<&'a str, usize> = HashMap::new();
            for start in positions {
                let end = start + *len;
                if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                    continue;
                }
                let substring = &text[start..end];
                if !eligible_substring(substring) {
                    continue;
                }
                *exact.entry(substring).or_insert(0) += 1;
            }

            for (substring, count) in exact {
                if count >= DICTIONARY_MIN_OCCURRENCES {
                    let entry = counts.entry(substring).or_insert(0);
                    *entry = (*entry).max(count);
                }
            }
        }
    }

    let mut candidates = counts
        .into_iter()
        .filter_map(|(original, count)| {
            if count < DICTIONARY_MIN_OCCURRENCES {
                return None;
            }
            let marker = marker_for(nonce, 0);
            let savings = estimated_savings(original, count, marker.len());
            (savings > 0).then_some(Candidate {
                original,
                count,
                savings,
            })
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| {
        b.savings
            .cmp(&a.savings)
            .then_with(|| b.original.len().cmp(&a.original.len()))
            .then_with(|| b.count.cmp(&a.count))
            .then_with(|| a.original.cmp(b.original))
    });
    candidates
}

fn collect_rolling_hash_positions(
    bytes: &[u8],
    window_len: usize,
    positions: &mut HashMap<u64, Vec<usize>>,
) {
    if window_len == 0 || window_len > bytes.len() {
        return;
    }

    let mut hash = 0u64;
    let mut power = 1u64;
    for (idx, byte) in bytes.iter().take(window_len).enumerate() {
        hash = hash
            .wrapping_mul(ROLLING_HASH_BASE)
            .wrapping_add(*byte as u64);
        if idx + 1 < window_len {
            power = power.wrapping_mul(ROLLING_HASH_BASE);
        }
    }
    positions.entry(hash).or_default().push(0);

    for start in 1..=bytes.len() - window_len {
        let outgoing = (bytes[start - 1] as u64).wrapping_mul(power);
        hash = hash
            .wrapping_sub(outgoing)
            .wrapping_mul(ROLLING_HASH_BASE)
            .wrapping_add(bytes[start + window_len - 1] as u64);
        positions.entry(hash).or_default().push(start);
    }
}

fn select_non_overlapping<'a>(
    text: &'a str,
    candidates: &[Candidate<'a>],
    nonce: &str,
) -> Vec<SelectedEntry<'a>> {
    // Lazy-greedy selection: start with stale estimated savings; re-evaluate
    // realized savings (accounting for used positions) before committing. If
    // realized savings fall below the next candidate's current estimate,
    // reinsert at the realized value so a better candidate can be selected
    // first, bounded by MAX_REEVALS to keep the loop terminating.
    let mut heap = BinaryHeap::<(isize, usize)>::with_capacity(candidates.len());
    for (idx, c) in candidates.iter().enumerate() {
        heap.push((c.savings, idx));
    }

    let mut selected = Vec::new();
    let mut used = vec![false; text.len()];
    let mut reeval_count = 0usize;
    const MAX_REEVALS: usize = 256;

    while let Some((estimate, idx)) = heap.pop() {
        if selected.len() >= DICTIONARY_MAX_DICT_SIZE {
            break;
        }
        let candidate = &candidates[idx];
        let marker = marker_for(nonce, selected.len() + 1);
        if text.contains(&marker) {
            continue;
        }
        let positions = non_overlapping_positions(text, candidate.original, &used);
        if positions.len() < DICTIONARY_MIN_OCCURRENCES {
            continue;
        }
        let realized = estimated_savings(candidate.original, positions.len(), marker.len());
        // C3: per-entry savings floor — skip entries that won't pay off.
        if realized < DICTIONARY_MIN_ENTRY_SAVINGS_BYTES as isize {
            continue;
        }
        // C2: if realized savings dropped below the next candidate's estimate
        // and we still have reeval budget, reinsert at the realized value.
        if realized < estimate {
            let next_best = heap.peek().map(|&(s, _)| s).unwrap_or(isize::MIN);
            if realized < next_best && reeval_count < MAX_REEVALS {
                heap.push((realized, idx));
                reeval_count += 1;
                continue;
            }
        }
        for pos in &positions {
            mark_range(&mut used, *pos, candidate.original.len());
        }
        selected.push(SelectedEntry {
            marker,
            original: candidate.original,
            positions,
        });
    }

    selected
}

fn non_overlapping_positions(text: &str, needle: &str, used: &[bool]) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut search_start = 0usize;
    while search_start < text.len() {
        let Some(relative) = text[search_start..].find(needle) else {
            break;
        };
        let pos = search_start + relative;
        if !range_overlaps(used, pos, needle.len()) {
            positions.push(pos);
            search_start = pos + needle.len();
        } else {
            search_start = advance_one_char(text, pos);
        }
    }
    positions
}

fn apply_replacements(text: &str, selected: &[SelectedEntry<'_>]) -> String {
    let mut replacements = selected
        .iter()
        .flat_map(|entry| {
            entry.positions.iter().map(|pos| Replacement {
                pos: *pos,
                len: entry.original.len(),
                marker: entry.marker.as_str(),
            })
        })
        .collect::<Vec<_>>();
    replacements.sort_by_key(|replacement| std::cmp::Reverse(replacement.pos));

    let mut compressed = text.to_string();
    for replacement in replacements {
        compressed.replace_range(
            replacement.pos..replacement.pos + replacement.len,
            replacement.marker,
        );
    }
    compressed
}

#[derive(Debug)]
struct Replacement<'a> {
    pos: usize,
    len: usize,
    marker: &'a str,
}

fn build_dictionary(selected: &[SelectedEntry<'_>]) -> Option<String> {
    let mut lines = Vec::with_capacity(selected.len() + 1);
    for entry in selected {
        let encoded = serde_json::to_string(entry.original).ok()?;
        lines.push(format!("{} = {encoded}", entry.marker));
    }
    lines.push(String::new());
    Some(lines.join("\n"))
}

fn slice_chars<'a>(text: &'a str, offsets: &[usize], start: usize, len: usize) -> Option<&'a str> {
    let end = start.checked_add(len)?;
    Some(&text[*offsets.get(start)?..*offsets.get(end)?])
}

fn eligible_substring(value: &str) -> bool {
    if value.contains('\n') || value.contains('\r') {
        return false;
    }
    !value
        .chars()
        .all(|ch| ch.is_whitespace() || matches!(ch, '-' | '_' | '='))
}

fn estimated_savings(original: &str, count: usize, marker_len: usize) -> isize {
    let original_cost = count.saturating_mul(original.len());
    let replacement_cost = count
        .saturating_mul(marker_len)
        .saturating_add(dictionary_entry_len(original, marker_len));
    original_cost as isize - replacement_cost as isize
}

fn dictionary_entry_len(original: &str, marker_len: usize) -> usize {
    let encoded_len = serde_json::to_string(original)
        .map(|encoded| encoded.len())
        .unwrap_or(original.len() + 2);
    marker_len + " = ".len() + encoded_len + "\n".len()
}

fn marker_for(nonce: &str, index: usize) -> String {
    format!("<<F{nonce}:{index}>>")
}

fn collision_free_nonce(text: &str) -> Option<String> {
    for nonce in 1..=999usize {
        let prefix = format!("<<F{nonce}:");
        if !text.contains(&prefix) {
            return Some(nonce.to_string());
        }
    }
    None
}

fn range_overlaps(used: &[bool], pos: usize, len: usize) -> bool {
    used.get(pos..pos + len)
        .is_none_or(|range| range.iter().any(|used| *used))
}

fn mark_range(used: &mut [bool], pos: usize, len: usize) {
    if let Some(range) = used.get_mut(pos..pos + len) {
        for slot in range {
            *slot = true;
        }
    }
}

fn advance_one_char(text: &str, pos: usize) -> usize {
    let mut chars = text[pos..].char_indices();
    let _ = chars.next();
    chars.next().map(|(idx, _)| pos + idx).unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_output() -> String {
        (0..24)
            .map(|_| "error: repeated dependency resolution failure in workspace crate alpha\n")
            .collect::<String>()
    }

    fn decompress_lzw_dictionary(output: &str) -> String {
        let Some(rest) = output.strip_prefix(DICTIONARY_HEADER) else {
            return output.to_string();
        };
        let Some(rest) = rest.strip_prefix('\n') else {
            return output.to_string();
        };
        let Some((dict, body)) = rest.split_once("\n\n") else {
            return output.to_string();
        };

        let mut result = body.to_string();
        for line in dict.lines().rev() {
            let Some((marker, encoded)) = line.split_once(" = ") else {
                continue;
            };
            let Ok(original) = serde_json::from_str::<String>(encoded) else {
                continue;
            };
            result = result.replace(marker, &original);
        }
        result
    }

    #[test]
    fn aggressive_lzw_compresses_repetitive_output() {
        let raw = repeated_output();
        let compressed = compress_lzw_dictionary(&raw).expect("compressible");

        assert!(compressed.starts_with(DICTIONARY_HEADER));
        assert!(compressed.len() < raw.len());
    }

    #[test]
    fn aggressive_lzw_skips_non_repetitive_output() {
        let raw = pseudo_random_ascii(2_000);

        assert_eq!(compress_lzw_dictionary(&raw), None);
    }

    #[test]
    fn aggressive_lzw_skips_oversized_output() {
        let raw = "error: repeated dependency resolution failure "
            .repeat(DICTIONARY_MAX_INPUT_BYTES / 10)
            + "tail";

        assert_eq!(compress_lzw_dictionary(&raw), None);
    }

    #[test]
    fn aggressive_lzw_handles_marker_collisions() {
        let raw = format!("<<F1:1>>\n{}", repeated_output());
        let compressed = compress_lzw_dictionary(&raw).expect("compressible");

        assert!(compressed.contains("<<F2:"));
        assert_eq!(decompress_lzw_dictionary(&compressed), raw);
    }

    #[test]
    fn aggressive_lzw_round_trips_with_test_decompressor() {
        let raw = repeated_output();
        let compressed = compress_lzw_dictionary(&raw).expect("compressible");

        assert_eq!(decompress_lzw_dictionary(&compressed), raw);
    }

    #[test]
    fn aggressive_lzw_skips_already_compressed_output() {
        let raw = format!("{DICTIONARY_HEADER}\n<<F1:1>> = \"value\"\n\nbody");

        assert_eq!(compress_lzw_dictionary(&raw), None);
    }

    #[test]
    fn aggressive_lzw_skips_repair_dictionary_output() {
        let raw = format!(
            "{}\n<<R1:1>> = \"value\"\n\nbody",
            super::super::REPAIR_DICTIONARY_HEADER
        );

        assert_eq!(compress_lzw_dictionary(&raw), None);
    }

    fn pseudo_random_ascii(len: usize) -> String {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut output = String::with_capacity(len);
        for _ in 0..len {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let byte = b'a' + ((seed >> 32) % 26) as u8;
            output.push(byte as char);
        }
        output
    }
}
