from __future__ import annotations

import uuid

from agenteval.schema.test_case import TestCase
from agenteval.simulation.agent_client import AgentClient
from agenteval.simulation.session import Session, Turn
from agenteval.simulation.simulator_factory import create_simulator


def _parse_termination(response_text: str) -> tuple[str, str | None]:
    lines = response_text.strip().split("\n")
    last_line = lines[-1].strip()
    normalised = last_line.strip("`\"'.,!? ").strip()

    if normalised == "[GOAL_ACHIEVED]":
        clean = "\n".join(lines[:-1]).strip()
        return clean, "goal_achieved"

    if "[GOAL_ACHIEVED]" in response_text:
        clean = response_text.replace("[GOAL_ACHIEVED]", "").strip()
        return clean, "goal_achieved"

    return response_text, None


async def run_single_scenario(test_case: TestCase, mode: str) -> Session:
    session_id = str(uuid.uuid4())
    agent_client = AgentClient(test_case.agent)
    simulator = create_simulator(mode, test_case)
    turns: list[Turn] = []
    turn_number = 1
    termination_reason = "max_turns_reached"

    opening_message = simulator.get_opening_message()
    turns.append(Turn(number=turn_number, role="user", content=opening_message))
    turn_number += 1

    try:
        last_agent_reply = await agent_client.send(session_id, opening_message)
    except Exception as error:
        turns.append(
            Turn(number=turn_number, role="agent", content=f"ERROR: {str(error)}")
        )
        session = Session(
            scenario_id=test_case.meta.scenario_id,
            session_id=session_id,
            turns=turns,
            turns_used=0,
            termination_reason="agent_error",
        )
        session.turns_used = len(session.user_turns())
        return session

    turns.append(Turn(number=turn_number, role="agent", content=last_agent_reply))
    turn_number += 1

    turn_index = 0
    # -1 because the opening exchange already happened before the loop
    while turn_index < test_case.conversation.max_turns - 1:
        next_user_message = simulator.get_next_turn(turn_index, last_agent_reply)
        if next_user_message is None:
            termination_reason = "max_turns_reached"
            break

        clean_message, signal = _parse_termination(next_user_message)
        if clean_message:
            turns.append(Turn(number=turn_number, role="user", content=clean_message))
            turn_number += 1

        if signal == "goal_achieved":
            termination_reason = "goal_achieved"
            break

        try:
            last_agent_reply = await agent_client.send(session_id, clean_message)
        except Exception as error:
            turns.append(
                Turn(number=turn_number, role="agent", content=f"ERROR: {str(error)}")
            )
            termination_reason = "agent_error"
            break

        turns.append(Turn(number=turn_number, role="agent", content=last_agent_reply))
        turn_number += 1
        turn_index += 1

    session = Session(
        scenario_id=test_case.meta.scenario_id,
        session_id=session_id,
        turns=turns,
        turns_used=0,
        termination_reason=termination_reason,
    )
    session.turns_used = len(session.user_turns())
    return session
