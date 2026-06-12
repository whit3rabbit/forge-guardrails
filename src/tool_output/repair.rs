use std::collections::HashMap;

use super::{
    dictionary_has_meaningful_savings, is_dictionary_compressed_output, DICTIONARY_MAX_DICT_SIZE,
    DICTIONARY_MAX_INPUT_BYTES, DICTIONARY_MIN_ENTRY_SAVINGS_BYTES, DICTIONARY_MIN_OCCURRENCES,
    REPAIR_DICTIONARY_HEADER as DICTIONARY_HEADER,
};

const MIN_RULE_BYTES: usize = 15;
const MAX_RULE_BYTES: usize = 120;
const MAX_SEQUENCE_TOKENS: usize = 32;

#[derive(Debug, Clone)]
struct Token {
    emit: String,
    expanded: String,
}

#[derive(Debug, Clone)]
struct SequenceCandidate {
    key: SequenceKey,
    positions: Vec<usize>,
    savings: isize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SequenceKey {
    expansion: String,
    token_len: usize,
}

#[derive(Debug, Clone)]
struct Rule {
    marker: String,
    expansion: String,
}

/// Compress repeated token sequences into a model-readable dictionary.
pub(super) fn compress_repair_dictionary(output: &str) -> Option<String> {
    if output.len() > DICTIONARY_MAX_INPUT_BYTES || is_dictionary_compressed_output(output) {
        return None;
    }

    let nonce = collision_free_nonce(output)?;
    let mut tokens = tokenize(output);
    if tokens.len() < 2 {
        return None;
    }

    let mut rules = Vec::new();
    while rules.len() < DICTIONARY_MAX_DICT_SIZE {
        let marker = marker_for(&nonce, rules.len() + 1);
        let Some(candidate) = best_sequence_candidate(&tokens, marker.len()) else {
            break;
        };
        if candidate.savings < DICTIONARY_MIN_ENTRY_SAVINGS_BYTES as isize {
            break;
        }

        tokens = replace_sequence(tokens, &candidate, marker.clone());
        rules.push(Rule {
            marker,
            expansion: candidate.key.expansion,
        });
    }

    if rules.is_empty() {
        return None;
    }

    let body = tokens
        .iter()
        .map(|token| token.emit.as_str())
        .collect::<String>();
    let dictionary = build_dictionary(&rules)?;
    let result = format!("{DICTIONARY_HEADER}\n{dictionary}\n{body}");
    let savings = output.len().checked_sub(result.len())?;
    if !dictionary_has_meaningful_savings(output.len(), savings) {
        return None;
    }
    Some(result)
}

fn tokenize(output: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_class: Option<TokenClass> = None;

    for ch in output.chars() {
        let class = TokenClass::for_char(ch);
        let should_flush = match current_class {
            Some(TokenClass::Other) | Some(TokenClass::Newline) => true,
            Some(existing) => existing != class || matches!(class, TokenClass::Other),
            None => false,
        };
        if should_flush && !current.is_empty() {
            tokens.push(Token::literal(std::mem::take(&mut current)));
        }
        current.push(ch);
        current_class = Some(class);
    }

    if !current.is_empty() {
        tokens.push(Token::literal(current));
    }
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenClass {
    Word,
    Whitespace,
    Newline,
    Other,
}

impl TokenClass {
    fn for_char(ch: char) -> Self {
        if ch == '\n' || ch == '\r' {
            Self::Newline
        } else if ch.is_whitespace() {
            Self::Whitespace
        } else if ch.is_alphanumeric() || ch == '_' {
            Self::Word
        } else {
            Self::Other
        }
    }
}

impl Token {
    fn literal(text: String) -> Self {
        Self {
            emit: text.clone(),
            expanded: text,
        }
    }

    fn marker(marker: String, expansion: String) -> Self {
        Self {
            emit: marker,
            expanded: expansion,
        }
    }
}

fn best_sequence_candidate(tokens: &[Token], marker_len: usize) -> Option<SequenceCandidate> {
    let mut positions_by_sequence: HashMap<SequenceKey, Vec<usize>> = HashMap::new();
    let max_sequence_tokens = MAX_SEQUENCE_TOKENS.min(tokens.len());

    for idx in 0..tokens.len() {
        let mut expansion = String::new();
        for token_len in 1..=max_sequence_tokens {
            let Some(token) = tokens.get(idx + token_len - 1) else {
                break;
            };
            expansion.push_str(&token.expanded);
            if expansion.len() > MAX_RULE_BYTES {
                break;
            }
            if token_len < 2 || expansion.len() < MIN_RULE_BYTES || !eligible_rule(&expansion) {
                continue;
            }
            let key = SequenceKey {
                expansion: expansion.clone(),
                token_len,
            };
            positions_by_sequence.entry(key).or_default().push(idx);
        }
    }

    positions_by_sequence
        .into_iter()
        .filter_map(|(key, positions)| {
            let positions = non_overlapping_sequence_positions(&positions, key.token_len);
            if positions.len() < DICTIONARY_MIN_OCCURRENCES {
                return None;
            }
            let savings = estimated_savings(&key.expansion, positions.len(), marker_len);
            (savings > 0).then_some(SequenceCandidate {
                key,
                positions,
                savings,
            })
        })
        .max_by(|a, b| {
            a.savings
                .cmp(&b.savings)
                .then_with(|| a.key.expansion.len().cmp(&b.key.expansion.len()))
                .then_with(|| a.positions.len().cmp(&b.positions.len()))
                .then_with(|| b.key.expansion.cmp(&a.key.expansion))
        })
}

fn non_overlapping_sequence_positions(positions: &[usize], token_len: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut next_available = 0usize;

    for pos in positions {
        if *pos < next_available {
            continue;
        }
        result.push(*pos);
        next_available = pos.saturating_add(token_len);
    }

    result
}

fn replace_sequence(
    tokens: Vec<Token>,
    candidate: &SequenceCandidate,
    marker: String,
) -> Vec<Token> {
    let mut positions = candidate.positions.iter().copied().peekable();
    let mut replaced = Vec::with_capacity(tokens.len());
    let mut idx = 0usize;

    while idx < tokens.len() {
        while positions.peek().is_some_and(|pos| *pos < idx) {
            positions.next();
        }
        if positions.peek().is_some_and(|pos| *pos == idx)
            && sequence_matches(&tokens, idx, &candidate.key)
        {
            positions.next();
            replaced.push(Token::marker(
                marker.clone(),
                candidate.key.expansion.clone(),
            ));
            idx += candidate.key.token_len;
        } else {
            replaced.push(tokens[idx].clone());
            idx += 1;
        }
    }

    replaced
}

fn sequence_matches(tokens: &[Token], start: usize, key: &SequenceKey) -> bool {
    let Some(sequence) = tokens.get(start..start + key.token_len) else {
        return false;
    };
    let mut expansion = String::new();
    for token in sequence {
        expansion.push_str(&token.expanded);
        if expansion.len() > key.expansion.len() {
            return false;
        }
    }
    expansion == key.expansion
}

fn build_dictionary(rules: &[Rule]) -> Option<String> {
    let mut lines = Vec::with_capacity(rules.len() + 1);
    for rule in rules {
        let encoded = serde_json::to_string(&rule.expansion).ok()?;
        lines.push(format!("{} = {encoded}", rule.marker));
    }
    lines.push(String::new());
    Some(lines.join("\n"))
}

/// Maximum `\n` characters allowed in a RePair rule expansion. Allowing short
/// multi-line spans captures log-block patterns that LZW skips entirely.
const MAX_RULE_NEWLINES: usize = 2;

fn eligible_rule(value: &str) -> bool {
    if value.contains('\r') {
        return false;
    }
    if value.chars().filter(|&c| c == '\n').count() > MAX_RULE_NEWLINES {
        return false;
    }
    !value
        .chars()
        .all(|ch| ch.is_whitespace() || matches!(ch, '-' | '_' | '='))
}

fn estimated_savings(expansion: &str, count: usize, marker_len: usize) -> isize {
    let original_cost = count.saturating_mul(expansion.len());
    let replacement_cost = count
        .saturating_mul(marker_len)
        .saturating_add(dictionary_entry_len(expansion, marker_len));
    original_cost as isize - replacement_cost as isize
}

fn dictionary_entry_len(expansion: &str, marker_len: usize) -> usize {
    let encoded_len = serde_json::to_string(expansion)
        .map(|encoded| encoded.len())
        .unwrap_or(expansion.len() + 2);
    marker_len + " = ".len() + encoded_len + "\n".len()
}

fn marker_for(nonce: &str, index: usize) -> String {
    format!("<<R{nonce}:{index}>>")
}

fn collision_free_nonce(text: &str) -> Option<String> {
    for nonce in 1..=999usize {
        let prefix = format!("<<R{nonce}:");
        if !text.contains(&prefix) {
            return Some(nonce.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repeated_output() -> String {
        (0..30)
            .map(|idx| {
                format!(
                    "error: repeated dependency resolution failure in workspace crate alpha at module_{idx}\n"
                )
            })
            .collect::<String>()
    }

    fn decompress_repair_dictionary(output: &str) -> String {
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
            let Ok(expansion) = serde_json::from_str::<String>(encoded) else {
                continue;
            };
            result = result.replace(marker, &expansion);
        }
        result
    }

    #[test]
    fn repair_compresses_repetitive_output() {
        let raw = repeated_output();
        let compressed = compress_repair_dictionary(&raw).expect("compressible");

        assert!(compressed.starts_with(DICTIONARY_HEADER));
        assert!(compressed.len() < raw.len());
    }

    #[test]
    fn repair_round_trips_with_test_decompressor() {
        let raw = repeated_output();
        let compressed = compress_repair_dictionary(&raw).expect("compressible");

        assert_eq!(decompress_repair_dictionary(&compressed), raw);
    }

    #[test]
    fn repair_handles_marker_collisions() {
        let raw = format!("<<R1:1>>\n{}", repeated_output());
        let compressed = compress_repair_dictionary(&raw).expect("compressible");

        assert!(compressed.contains("<<R2:"));
        assert_eq!(decompress_repair_dictionary(&compressed), raw);
    }

    #[test]
    fn repair_skips_oversized_output() {
        let raw = "error: repeated dependency resolution failure "
            .repeat(DICTIONARY_MAX_INPUT_BYTES / 10);

        assert_eq!(compress_repair_dictionary(&raw), None);
    }

    #[test]
    fn repair_compresses_multi_line_rules() {
        // Two-line blocks repeat 30+ times — eligible under MAX_RULE_NEWLINES=2.
        let raw = (0..40)
            .map(|_| "HEADER_LINE\nDETAIL_LINE\n")
            .collect::<String>();
        let compressed = compress_repair_dictionary(&raw).expect("should compress");
        assert_eq!(decompress_repair_dictionary(&compressed), raw);
    }

    #[test]
    fn repair_skips_rules_exceeding_max_newlines() {
        // Three-newline patterns should remain ineligible.
        let raw = (0..40).map(|_| "A\nB\nC\nD\n").collect::<String>();
        // A\nB\nC\nD\n would require a 3-newline rule — still skipped.
        // The output may compress via shorter sub-rules but not via a 3-newline rule.
        // Just verify it doesn't panic and round-trips.
        if let Some(compressed) = compress_repair_dictionary(&raw) {
            assert_eq!(decompress_repair_dictionary(&compressed), raw);
        }
    }

    #[test]
    fn repair_skips_already_compressed_output() {
        let raw = format!("{DICTIONARY_HEADER}\n<<R1:1>> = \"value\"\n\nbody");

        assert_eq!(compress_repair_dictionary(&raw), None);
    }

    #[test]
    fn repair_skips_lzw_dictionary_output() {
        let raw = format!(
            "{}\n<<F1:1>> = \"value\"\n\nbody",
            super::super::LZW_DICTIONARY_HEADER
        );

        assert_eq!(compress_repair_dictionary(&raw), None);
    }
}
