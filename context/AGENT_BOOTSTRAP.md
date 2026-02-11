# Agent Bootstrap

Paste this snippet into any new chat (Copilot, JetBrains AI, ChatGPT) before asking for help.

---

Before proposing changes:

1) Read `context/PROJECT_CONTEXT.md`
2) Read `context/AGENT_PLAYBOOK.md` in this repository
3) Assume the following constraints:
   - Core C++ library built with CMake
   - R bindings via Rcpp; Python bindings via pybind11
   - ACTIONet packages/modules/libraries usually installed by standard means, but via conda isolation when used in pipeline

Rules of engagement:

- Prefer small, incremental diffs over large rewrites
- Do not change output formats, parameter names, or directory structures without explicitly calling it out and getting confirmation
- When uncertain about data contracts or expected behavior, ask for an example input/output or schema

When blocked, explicitly request:

- Example inputs (small, representative)
- Expected outputs (golden examples if available)
- Relevant logs or error messages

---
