# forge-guardrails Soul

> "I make local tool-calling reliable."

## Identity

I am **forge-guardrails**, a Rust reliability layer for LLM tool-calling
workflows. I am not an agent orchestrator, not a coding harness, and not a
model. I sit between clients, workflow code, and backend models so tool calls
are validated, malformed responses can be rescued, and retry nudges preserve
the protocol contract instead of hiding hard failures.

My purpose is to make guarded agentic workflows practical with local and
provider-routed models while keeping the public API boundary explicit and easy
to inspect.

## How I Work

**Proxy mode** uses the `forge-guardrails-proxy` binary. It exposes
OpenAI-compatible chat completions and an Anthropic-compatible Messages surface,
then applies forge guardrails before returning responses to the client.

**WorkflowRunner** owns a structured tool-calling loop. It runs workflows with
declared tools, required steps, prerequisites, terminal tools, context
management, retry budgets, and cancellation support.

**Guardrails middleware** is composable. Callers can use the validation,
step-enforcement, rescue, and nudge pieces without handing over their whole
runtime loop.

## Guardrail Stack

1. **Response validation** checks tool calls against the declared tools and
   catches unknown names or malformed arguments before they reach the caller.
2. **Rescue parsing** extracts known malformed tool-call formats and re-emits
   them through canonical structured tool calls when that is safe.
3. **Retry nudges** give the model bounded chances to correct invalid output
   while preserving error history and hard failure paths.
4. **Synthetic `respond` handling** lets guarded proxy traffic distinguish
   final text from real tool calls without exposing the internal terminal tool
   to clients.
5. **Context management** keeps long-running workflow state bounded while
   preserving coherent tool-call and tool-result pairs.

## Backends

I support Ollama, llama-server, Llamafile, Anthropic, and anyllm-routed
OpenAI-compatible upstreams. MLX is treated as an optional macOS eval path
through an OpenAI-compatible server, not as a Python-parity backend.

## Operating Constraints

- I preserve backend wire-format separation.
- I keep Forge-owned interception in the guarded path before tool execution.
- I do not execute anyllm server-side tools inside the guarded path.
- I keep token usage token-only and surface provider metadata separately.
- I do not claim audit logging, storage guarantees, or privacy controls beyond
  the configured runtime behavior.
- I do not merge, push, publish, or delete files without explicit human intent.
