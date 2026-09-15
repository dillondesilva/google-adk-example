"""
Minimal Google ADK example showing where each Graph Engineering view
(Task Organization / Agent Coordination / Runtime State) actually lives
in the code.

Pipeline: two researchers run in parallel -> a drafter merges their
findings -> a critic reviews the draft in a loop until it passes.
"""

from typing import Literal

from google.adk import Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.workflow import JoinNode
from google.genai import types
from pydantic import BaseModel

MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# Agent coordination graph: who exists and what they can do. These are nodes.
# Nothing here says *when* they run -- that's the edges= list below.
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


class Review(BaseModel):
    verdict: Literal["approved", "revise"]
    feedback: str = ""


critic = LlmAgent(
    name="critic",
    model=MODEL,
    instruction=(
        "Review {draft}. If it is clear and well-supported, verdict=approved. "
        "Otherwise verdict=revise and write one sentence of feedback."
    ),
    output_schema=Review,
)


# Workflow loops need a routed Event -- unconditional cycles are rejected.
# `route` is the edge label; `state` writes feedback for the next draft.
def route(node_input: Review):
    return Event(
        route=node_input.verdict,
        state={"feedback": node_input.feedback},
        message=node_input.verdict,
    )


# ---------------------------------------------------------------------------
# Task organization graph: the edges. Nested tuple = parallel fan-out,
# JoinNode = fan-in, dict = conditional loop. Swap drafter for a different
# agent and this structure does not change.
#
#   START -> (researcher_a || researcher_b) -> join -> drafter -> critic -> route
#              ^                                                        |
#              +---------------------- "revise" ------------------------+
#                                       "approved" ends the run
# ---------------------------------------------------------------------------
join = JoinNode(name="join")

report_pipeline = Workflow(
    name="report_pipeline",
    edges=[
        ("START", (researcher_a, researcher_b), join, drafter, critic, route),
        (route, {"revise": drafter}),
    ],
)


# ---------------------------------------------------------------------------
# Runtime state: after a run, session.state is the record of what happened --
# research_a, research_b, draft, feedback. Written by different nodes at
# different times. This is the coupling layer between the two graphs above.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    from google.adk.runners import InMemoryRunner

    load_dotenv()

    runner = InMemoryRunner(node=report_pipeline, app_name="graph_eng_demo")
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
        state_delta={"topic": topic},
    ):
        if event.content and event.content.parts and event.content.parts[0].text:
            print(f"[{event.author}] {event.content.parts[0].text}")

    final_state = runner.session_service.get_session_sync(
        app_name="graph_eng_demo", user_id="demo_user", session_id=session.id
    ).state
    print("\n--- runtime state ---")
    for key, value in final_state.items():
        print(f"{key}: {value}")
