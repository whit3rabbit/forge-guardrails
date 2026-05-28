SHELL := /usr/bin/env bash

CARGO ?= cargo
EVAL_SCRIPT ?= scripts/run_local_eval.sh

FEATURES ?= classifier
FEATURE_FLAGS := $(if $(strip $(FEATURES)),--features $(FEATURES),)

RUNS ?= 10
SUITE ?= release
CLASSIFIER_MODE ?= shadow
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

.PHONY: help build check test fmt fmt-check clippy eval eval-release eval-release-classify eval-smoke eval-smoke-classify

help:
	@printf '%s\n' 'Targets:'
	@printf '  %-24s %s\n' 'build' 'cargo build with FEATURES, default classifier'
	@printf '  %-24s %s\n' 'check' 'cargo check with FEATURES'
	@printf '  %-24s %s\n' 'test' 'cargo test with FEATURES'
	@printf '  %-24s %s\n' 'fmt-check' 'cargo fmt --all --check'
	@printf '  %-24s %s\n' 'clippy' 'cargo clippy --all-targets with FEATURES'
	@printf '  %-24s %s\n' 'eval-release' '10-run release eval, no classifier, resource baseline'
	@printf '  %-24s %s\n' 'eval-release-classify' '10-run release eval, classifier, resource baseline'
	@printf '  %-24s %s\n' 'eval-smoke' 'smoke eval, no classifier, resource baseline'
	@printf '  %-24s %s\n' 'eval-smoke-classify' 'smoke eval, classifier, resource baseline'
	@printf '%s\n' ''
	@printf '%s\n' 'Common overrides: FEATURES=classifier RUNS=10 OUTPUT_DIR=target/local-eval/name'
	@printf '%s\n' '                  CLASSIFIER_MODE=shadow RESOURCE_INTERVAL=1.0 EVAL_ARGS="..."'

build:
	$(CARGO) build $(FEATURE_FLAGS) --all-targets

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

eval-release-classify:
	$(EVAL_SCRIPT) $(EVAL_COMMON_ARGS) --classify --classifier-mode "$(CLASSIFIER_MODE)"

eval-smoke:
	$(MAKE) eval-release SUITE=smoke RUNS=1

eval-smoke-classify:
	$(MAKE) eval-release-classify SUITE=smoke RUNS=1
