# Google ADK graph-engineering demo

A small [Google ADK](https://google.github.io/adk-docs/) example that shows where **Graph Engineering** actually lives in code.

The [paper](https://arxiv.org/html/2608.21156v2) argues that multi-agent systems need more than a smarter single loop. They need explicit graphs for three things:

1. **Task organization** — how work is decomposed, ordered, parallelized, and verified
2. **Agent coordination** — who exists, what they can do, and how they relate
3. **Runtime state** — the evolving record of what happened, which later steps can read

This repo is not a framework. It is one short pipeline so those three views can be read off a single file.

## The pipeline

Two researchers run in parallel, a drafter merges their findings, and a critic reviews the draft until it passes (or the critic approves):

```
START -> (researcher_a || researcher_b) -> join -> drafter -> critic -> route
           ^                                                        |
           +---------------------- "revise" ------------------------+
                                    "approved" ends the run
```

All of that is in [`graph_eng_demo.py`](graph_eng_demo.py):

| Paper view | In this file |
| --- | --- |
| Agent coordination | The `LlmAgent` definitions — nodes, instructions, `output_key` |
| Task organization | `Workflow(edges=[...])` — fan-out, `JoinNode`, the `revise` back-edge |
| Runtime state | `session.state` after the run (`research_a`, `research_b`, `draft`, `feedback`) |

The agents could be swapped without changing the edge list. The edge list could change without rewriting who the agents are. State is the coupling layer between those two graphs.

## Run

You need a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
echo 'GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE' > .env

uv sync
uv run python graph_eng_demo.py
```

Each agent prints as it runs. The final block is `session.state` — the runtime state graph for that invocation.

## Reference

Yuyuan Feng et al., [*Graph Engineering in the Era of LLM Agents: From Individual Intelligence to System Intelligence*](https://arxiv.org/html/2608.21156v2), arXiv:2608.21156v2, 2026.
