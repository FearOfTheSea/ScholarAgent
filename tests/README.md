# Tests

Tests are organized by architectural boundary. They cover domain rules,
application use cases, local adapter persistence, API error behavior, loopback
startup, and the committed English-PDF retrieval corpus.

`test_adaptive_tutor_e2e.py` is gated by `RUN_LOCAL_E2E=1` and marked
`local_runtime`. It exercises the complete tutor API journey against the
bundled linear-regression lecture using real local embeddings and the selected
local LLM provider.

`test_study_agent_e2e.py` uses the same gate and exercises every capability in
the unified Study Agent catalog through one real, multi-action request. Set
`E2E_LLM_PROVIDER=scratch_gpt` to run that journey through ScholarGPT instead
of Ollama.
