"""
Generates synthetic long-context transcripts for the NOC agent, each burying one
critical detail early under 25-35 turns of tool-output noise, then asking a final
question that requires that detail to answer correctly.

Mirrors the lab's worked example (allergy detail buried under tool JSON) but for
Nexlink: a VIP SLA / compliance detail mentioned once early, then the agent has to
recall it dozens of turns later when a bandwidth decision is being made.
"""
import random
from typing import List, Dict, Any


NOISE_NODES = ["Node #10 (Alex-Fiber-Main)", "Node #11 (Cairo-Fiber-East)",
               "Node #12 (Alex-Fiber-North)", "Node #13 (Suez-5G-Core)"]
NOISE_METRICS = [
    "Packet loss 0.02%, jitter 3ms, latency 12ms.",
    "Uptime 99.98% over trailing 30 days.",
    "Optical power -14.2 dBm, within nominal range.",
    "CPU utilization on edge router: 34%.",
    "No anomalies detected in the last polling cycle.",
    "Routine health check passed, no action needed.",
]

CRITICAL_DETAILS = [
    "Customer 405 (Bank of Alexandria) has a VIP SLA clause: any bandwidth change above "
    "3.0 Gbps requires NOC_Admin sign-off AND a 24-hour advance compliance notice due to "
    "banking regulatory requirements.",
    "Customer 407 (Alexandria Hospital) is on a VIP SLA with a hard requirement: any node "
    "serving this customer must failover within 90 seconds during an outage, per healthcare "
    "continuity regulation.",
    "Node #12 (Alex-Fiber-North) had a prior unresolved firmware bug in its optical "
    "transceiver (batch QX-771) that causes intermittent Down status under thermal load "
    "above 35C, flagged by vendor recall notice.",
    "Customer 408 (Delta Logistics) is Standard tier but has a signed contractual penalty "
    "clause: any Suspended status lasting over 4 hours triggers automatic SLA credit "
    "reimbursement of 15% of monthly fees.",
]

FINAL_QUESTIONS = [
    "Before we approve this bandwidth change for customer 405, are there any compliance "
    "concerns we should flag?",
    "If node 12 goes into maintenance for cooling work, what should we watch for based on "
    "past history?",
    "Customer 407 just reported a brief outage -- did we meet the failover requirement?",
    "Customer 408's service has been suspended for 5 hours -- any contractual exposure?",
]


def _noise_turn(rng: random.Random) -> Dict[str, Any]:
    node = rng.choice(NOISE_NODES)
    metric = rng.choice(NOISE_METRICS)
    return {"role": "tool", "type": "tool_result", "content": f"Diagnostic {node}: {metric}"}


def generate_transcript(variation_id: int, noise_turns: int = 32) -> Dict[str, Any]:
    """
    Returns {"turns": [...], "critical_detail": str, "final_question": str}
    critical_detail is planted at turn ~2-4, then noise_turns of tool bloat follow,
    then the final question turn is appended.
    """
    rng = random.Random(variation_id)
    idx = variation_id % len(CRITICAL_DETAILS)
    critical_detail = CRITICAL_DETAILS[idx]
    final_question = FINAL_QUESTIONS[idx]

    turns: List[Dict[str, Any]] = []
    turns.append({"role": "user", "type": "chat", "content": "Starting shift, pulling up dashboard."})
    turns.append({"role": "assistant", "type": "chat", "content": "Dashboard loaded, all nodes visible."})
    # Plant the critical detail early (turn index 2-4)
    plant_at = rng.randint(2, 4)
    while len(turns) < plant_at:
        turns.append(_noise_turn(rng))
    turns.append({"role": "tool", "type": "tool_result", "content": critical_detail})

    # Bury it under tool-heavy noise
    for _ in range(noise_turns):
        turns.append(_noise_turn(rng))
        if rng.random() < 0.15:
            turns.append({"role": "user", "type": "chat", "content": rng.choice(
                ["ok", "got it", "thanks", "checking next node", "moving on"]
            )})

    turns.append({"role": "user", "type": "chat", "content": final_question})

    return {"turns": turns, "critical_detail": critical_detail, "final_question": final_question}


def generate_suite(n_variations: int = 8, noise_turns: int = 32) -> List[Dict[str, Any]]:
    return [generate_transcript(i, noise_turns=noise_turns) for i in range(n_variations)]
