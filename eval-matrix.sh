MODEL="mistralai/mixtral-8x22b-instruct"
BASE="target/local-eval/or-mixtral"

common=(
  --suite release
  --runs 10
  --upstream-backend openrouter
  --model "$MODEL"
  --skip-published-compare
)

# 1. Baseline: no classifier, no final-response verifier, no compression
scripts/run_local_eval.sh "${common[@]}" \
  --tool-output-compression disabled \
  --output-dir "$BASE/baseline"

# 2. Tool-call classifier in shadow mode
scripts/run_local_eval.sh "${common[@]}" \
  --classify \
  --classifier-mode shadow \
  --tool-output-compression disabled \
  --output-dir "$BASE/toolcall-shadow"

# 3. Tool-call classifier + final-response verifier, both shadow
scripts/run_local_eval.sh "${common[@]}" \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --tool-output-compression disabled \
  --output-dir "$BASE/toolcall-final-shadow"

# 4. Compression only, no classifier/verifier
scripts/run_local_eval.sh "${common[@]}" \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto \
  --output-dir "$BASE/compression-aggressive"

# 5. Full stack: classifier + final verifier + aggressive compression
scripts/run_local_eval.sh "${common[@]}" \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto \
  --output-dir "$BASE/full-shadow-aggressive"