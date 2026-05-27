# Global Source Guidelines

When modifying files under the [src](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src) directory, adhere to the following top-level rules:

## Core Invariants

1. **Parity First**: The gold standard for reference behavior is the [forge](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge) submodule (Python). Avoid adding custom optimizations or behavior deviations without explicit parity tests or fixtures in [tests/parity](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/tests/parity).
2. **Behavioral Integrity**: Keep changes small and behavior-driven. Do not change existing public interfaces in [lib.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/lib.rs) without checking for broken imports in tests and binaries.
3. **No Placeholders**: Never write placeholder code or omit details. Implement full logic for any stubbed areas.
4. **Documentation**: Maintain documentation comments (`///`) on all public structures, traits, enums, functions, and modules. Keep them in sync with [docs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/docs).

## Validation Command Sequence

Before concluding any work or committing changes, run this sequence to ensure zero regressions:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```
