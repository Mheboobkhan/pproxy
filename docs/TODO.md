# Research TODO

Hypotheses were originally outlined in PREREG.md and are collected below. The current baseline and
local abstraction/rephrase modes provide the starting implementation for H1;
the hypotheses still need evaluation.

- [ ] **H1 — Prompt abstraction:** evaluate sensitive-detail removal and answer
  utility for the implemented local rephrase mode against baseline.
- [ ] **H2 — Approach based on the Shokari paper:** identify the exact paper,
  define the transformation and comparison criteria, and implement the mode.
- [ ] **H3 — Prompt chunking and response synthesis:** implement chunking,
  record chunk boundaries, and synthesize the cloud responses. The existing
  optional local synthesis step currently handles a single cloud response.
- [ ] **H4 — K-1 pseudo conversation:** define the conversation construction
  and evaluation procedure, then implement it as a selectable mode.
