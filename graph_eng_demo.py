"""
Minimal Google ADK example showing where each Graph Engineering view
(Task Organization / Agent Coordination / Runtime State) actually lives
in the code.

Pipeline: two researchers run in parallel -> a drafter merges their
findings -> a critic reviews the draft in a loop until it passes.
"""

from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# Runtime state: the "exit_loop" tool is how the Critic writes a signal into
# the shared runtime state that the LoopAgent itself reads to decide whether
# to keep iterating. This is the mechanism, not the record -- ADK's actual
# runtime state graph is session.state (the output_key writes below) plus
# the event history, which every agent shares.
# ---------------------------------------------------------------------------
def exit_loop(tool_context: ToolContext):
    """Call this once the draft passes review."""
    tool_context.actions.escalate = True
    return {"status": "approved"}


# ---------------------------------------------------------------------------
# Agent coordination graph: who exists, what they can do, and who they are
# relative to each other. Nothing here says *when* they run -- that's the
# task organization graph below. Capability = the tools/instruction each
# agent gets; team = the sub_agents nesting a few lines down.
# ---------------------------------------------------------------------------
researcher_a = LlmAgent(
    name="researcher_a",
    model=MODEL,
    instruction="Research the market size for {topic}. Be concise.",
    output_key="research_a",          # writes into session.state["research_a"]
)

researcher_b = LlmAgent(
    name="researcher_b",
    model=MODEL,
    instruction="Research the main competitors for {topic}. Be concise.",
    output_key="research_b",
)

drafter = LlmAgent(
    name="drafter",
    model=MODEL,
    instruction=(
        "Write a short report on {topic} using this research:\n"
        "Market size: {research_a}\n"
        "Competitors: {research_b}\n"
        "If {feedback?} was provided, revise the previous draft to address it."
    ),
    output_key="draft",
)

critic = LlmAgent(
    name="critic",
    model=MODEL,
    instruction=(
        "Review {draft}. If it is clear and well-supported, call exit_loop. "
        "Otherwise, write one sentence of feedback."
    ),
    tools=[exit_loop],
    output_key="feedback",
)


# ---------------------------------------------------------------------------
# Task organization graph: the actual dependency structure -- what runs in
# parallel, what runs in sequence, and what loops. This is a *different*
# object from the agent coordination graph above: swap drafter for a
# different agent and this structure doesn't change at all.
# ---------------------------------------------------------------------------
research_phase = ParallelAgent(
    name="research_phase",
    sub_agents=[researcher_a, researcher_b],
)

review_loop = LoopAgent(
    name="review_loop",
    sub_agents=[drafter, critic],
    max_iterations=3,
)

report_pipeline = SequentialAgent(
    name="report_pipeline",
    sub_agents=[research_phase, review_loop],
)


# ---------------------------------------------------------------------------
# Runtime state, made visible: after a run, session.state is the queryable
# record of what actually happened -- research_a, research_b, draft, and
# feedback all live here, written by different agents at different times.
# This is the coupling layer between the two graphs above.
# ---------------------------------------------------------------------------
def main():
    from dotenv import load_dotenv
    from google.adk.runners import InMemoryRunner

    load_dotenv()

    runner = InMemoryRunner(agent=report_pipeline, app_name="graph_eng_demo")
    session = runner.session_service.create_session_sync(
        app_name="graph_eng_demo", user_id="demo_user"
    )

    topic = "portable espresso machines"
    for event in runner.run(
        user_id="demo_user",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=f"topic: {topic}")],
        ),
        # {topic} in instructions is filled from session.state, not the user
        # message text -- this is the runtime-state coupling layer.
        state_delta={"topic": topic},
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"[{event.author}] {event.content.parts[0].text}")

    final_state = runner.session_service.get_session_sync(
        app_name="graph_eng_demo", user_id="demo_user", session_id=session.id
    ).state
    print("\n--- runtime state graph ---")
    for key, value in final_state.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
