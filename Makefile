SHELL := /usr/bin/env bash

CARGO ?= cargo
EVAL_SCRIPT ?= scripts/run_local_eval.sh

FEATURES ?= classifier
FEATURE_FLAGS := $(if $(strip $(FEATURES)),--features $(FEATURES),)

RUNS ?= 10
SUITE ?= release
CLASSIFIER_MODE ?= shadow
CLASSIFIER_MODEL ?= quantized
FINAL_RESPONSE_CLASSIFIER_MODE ?= shadow
FINAL_RESPONSE_CLASSIFIER_MODEL ?= quantized
FINAL_RESPONSE_SHADOW_OUTPUT_DIR ?= target/local-eval/release-onnx-final-shadow
RESOURCE_INTERVAL ?= 1.0

MODEL ?=
GGUF ?=
OUTPUT_DIR ?=
PROXY_PORT ?=
BACKEND_PORT ?=
EVAL_ARGS ?=

MODEL_ARGS := $(if $(strip $(MODEL)),--model "$(MODEL)",)
GGUF_ARGS := $(if $(strip $(GGUF)),--gguf "$(GGUF)",)
OUTPUT_DIR_ARGS := $(if $(strip $(OUTPUT_DIR)),--output-dir "$(OUTPUT_DIR)",)
PROXY_PORT_ARGS := $(if $(strip $(PROXY_PORT)),--proxy-port "$(PROXY_PORT)",)
BACKEND_PORT_ARGS := $(if $(strip $(BACKEND_PORT)),--backend-port "$(BACKEND_PORT)",)
FINAL_RESPONSE_SHADOW_OUTPUT := $(if $(strip $(OUTPUT_DIR)),$(OUTPUT_DIR),$(FINAL_RESPONSE_SHADOW_OUTPUT_DIR))

EVAL_COMMON_ARGS := \
	--suite "$(SUITE)" \
	--runs "$(RUNS)" \
	--resource-baseline \
	--resource-interval "$(RESOURCE_INTERVAL)" \
	$(MODEL_ARGS) \
	$(GGUF_ARGS) \
	$(OUTPUT_DIR_ARGS) \
	$(PROXY_PORT_ARGS) \
	$(BACKEND_PORT_ARGS) \
	$(EVAL_ARGS)

.PHONY: help build build-release check test fmt fmt-check clippy eval eval-release eval-release-baseline eval-release-classify eval-release-classify-shadow eval-release-classify-advisory eval-release-classify-enforce eval-release-final-response eval-release-final-response-shadow eval-smoke eval-smoke-classify eval-smoke-final-response

help:
	@printf '%s\n' 'Targets:'
	@printf '  %-36s %s\n' 'build' 'cargo build with FEATURES, default classifier'
	@printf '  %-36s %s\n' 'build-release' 'cargo build --release with FEATURES'
	@printf '  %-36s %s\n' 'check' 'cargo check with FEATURES'
	@printf '  %-36s %s\n' 'test' 'cargo test with FEATURES'
	@printf '  %-36s %s\n' 'fmt-check' 'cargo fmt --all --check'
	@printf '  %-36s %s\n' 'clippy' 'cargo clippy --all-targets with FEATURES'
	@printf '  %-36s %s\n' 'eval-release' '10-run release eval, no classifier, resource baseline'
	@printf '  %-36s %s\n' 'eval-release-baseline' 'alias for eval-release'
	@printf '  %-36s %s\n' 'eval-release-classify' 'release eval, tool-call classifier with CLASSIFIER_MODE'
	@printf '  %-36s %s\n' 'eval-release-classify-shadow' 'release eval, tool-call classifier shadow'
	@printf '  %-36s %s\n' 'eval-release-classify-advisory' 'release eval, tool-call classifier advisory'
	@printf '  %-36s %s\n' 'eval-release-classify-enforce' 'release eval, tool-call classifier enforce'
	@printf '  %-36s %s\n' 'eval-release-final-response' 'release eval, tool-call classifier plus final-response verifier'
	@printf '  %-36s %s\n' 'eval-release-final-response-shadow' 'shadow final-response verifier eval with standard output dir'
	@printf '  %-36s %s\n' 'eval-smoke' 'smoke eval, no classifier, resource baseline'
	@printf '  %-36s %s\n' 'eval-smoke-classify' 'smoke eval, classifier, resource baseline'
	@printf '  %-36s %s\n' 'eval-smoke-final-response' 'smoke eval, classifier plus final-response verifier'
	@printf '%s\n' ''
	@printf '%s\n' 'Common overrides: FEATURES=classifier RUNS=10 OUTPUT_DIR=target/local-eval/name'
	@printf '%s\n' '                  CLASSIFIER_MODE=shadow FINAL_RESPONSE_CLASSIFIER_MODE=shadow RESOURCE_INTERVAL=1.0 EVAL_ARGS="..."'

build:
	$(CARGO) build $(FEATURE_FLAGS) --all-targets

build-release:
	$(CARGO) build --release $(FEATURE_FLAGS) --all-targets

check:
	$(CARGO) check $(FEATURE_FLAGS) --all-targets

test:
	$(CARGO) test $(FEATURE_FLAGS)

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all --check

clippy:
	$(CARGO) clippy $(FEATURE_FLAGS) --all-targets -- -D warnings

eval: eval-release

eval-release:
	$(EVAL_SCRIPT) $(EVAL_COMMON_ARGS)

eval-release-baseline: eval-release

eval-release-classify:
	$(EVAL_SCRIPT) $(EVAL_COMMON_ARGS) --classify --classifier-mode "$(CLASSIFIER_MODE)" --classifier-model "$(CLASSIFIER_MODEL)"

eval-release-classify-shadow:
	$(MAKE) eval-release-classify CLASSIFIER_MODE=shadow

eval-release-classify-advisory:
	$(MAKE) eval-release-classify CLASSIFIER_MODE=advisory

eval-release-classify-enforce:
	$(MAKE) eval-release-classify CLASSIFIER_MODE=enforce

eval-release-final-response:
	$(EVAL_SCRIPT) $(EVAL_COMMON_ARGS) --classify --classifier-mode "$(CLASSIFIER_MODE)" --classifier-model "$(CLASSIFIER_MODEL)" --verify-final-response --final-response-classifier-mode "$(FINAL_RESPONSE_CLASSIFIER_MODE)" --final-response-classifier-model "$(FINAL_RESPONSE_CLASSIFIER_MODEL)"

eval-release-final-response-shadow:
	$(MAKE) eval-release-final-response CLASSIFIER_MODE=shadow FINAL_RESPONSE_CLASSIFIER_MODE=shadow OUTPUT_DIR="$(FINAL_RESPONSE_SHADOW_OUTPUT)"

eval-smoke:
	$(MAKE) eval-release SUITE=smoke RUNS=1

eval-smoke-classify:
	$(MAKE) eval-release-classify SUITE=smoke RUNS=1

eval-smoke-final-response:
	$(MAKE) eval-release-final-response SUITE=smoke RUNS=1
