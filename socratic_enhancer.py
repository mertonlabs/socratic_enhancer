#!/usr/bin/env python3
"""Socratic Enhancer: Iteratively refine a spec using two Claude agents in a Socratic dialogue loop."""

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    query,
)

# ANSI helpers
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class Spinner:
    """Async spinner that runs in the background."""

    def __init__(self, label: str):
        self.label = label
        self._task: asyncio.Task | None = None
        self._start_time = 0.0

    async def _spin(self):
        i = 0
        try:
            while True:
                elapsed = time.monotonic() - self._start_time
                frame = SPINNER_FRAMES[i % len(SPINNER_FRAMES)]
                sys.stderr.write(f"{CLEAR_LINE}{DIM}{frame} {self.label} ({elapsed:.0f}s){RESET}")
                sys.stderr.flush()
                i += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            elapsed = time.monotonic() - self._start_time
            sys.stderr.write(f"{CLEAR_LINE}")
            sys.stderr.flush()
            return elapsed

    async def __aenter__(self):
        self._start_time = time.monotonic()
        self._task = asyncio.create_task(self._spin())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            self.elapsed = await self._task


def extract_text(message: AssistantMessage) -> str:
    """Extract concatenated text from an AssistantMessage's content blocks."""
    parts = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(parts)


def extract_revised_spec(text: str) -> str | None:
    """Extract content between <revised_spec> tags."""
    match = re.search(r"<revised_spec>(.*?)</revised_spec>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def run_questioner(
    spec_text: str, num_questions: int, model: str, effort: str | None
) -> tuple[str, float]:
    """Run an ephemeral Questioner agent. Returns (questions_text, cost)."""
    prompt = f"""Read the following spec document and generate exactly {num_questions} Socratic questions that probe for:
- Gaps in the specification
- Ambiguities or unclear requirements
- Edge cases not addressed
- Unstated assumptions
- Missing details or implementation concerns

Output each question on its own line, numbered (1., 2., etc.).

<spec>
{spec_text}
</spec>"""

    options = ClaudeAgentOptions(
        system_prompt="You are a critical analyst. Your job is to ask probing, insightful questions about specifications to surface gaps, ambiguities, and unstated assumptions. Ask only questions — do not answer them.",
        model=model,
        permission_mode="bypassPermissions",
        allowed_tools=[],
        max_turns=1,
        effort=effort,
    )

    response_parts = []
    cost = 0.0
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            response_parts.append(extract_text(message))
        elif isinstance(message, ResultMessage) and message.total_cost_usd:
            cost = message.total_cost_usd

    return "\n".join(response_parts), cost


def print_header(text: str):
    width = 60
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def print_step(label: str, detail: str = ""):
    suffix = f" {DIM}{detail}{RESET}" if detail else ""
    print(f"\n{BOLD}{label}{RESET}{suffix}")


async def run(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(f"Error: spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    spec_text = spec_path.read_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save v0
    v0_path = output_dir / "v0.md"
    v0_path.write_text(spec_text)

    questioner_model = args.questioner_model or args.model
    planner_model = args.planner_model or args.model

    print_header("Socratic Enhancer")
    print(f"  Spec:       {spec_path}")
    print(f"  Questions:  {args.questions}/iteration")
    print(f"  Iterations: {args.iterations}")
    print(f"  Questioner: {questioner_model}")
    print(f"  Planner:    {planner_model}")
    if args.effort:
        print(f"  Effort:     {args.effort}")
    print(f"  Output:     {output_dir}/")
    print(f"\n{DIM}Saved initial spec as v0.md{RESET}")

    total_cost = 0.0

    planner_options = ClaudeAgentOptions(
        system_prompt=(
            "You are a spec refinement agent. You will be given a spec document and asked "
            "probing questions about it. Answer each question thoroughly, thinking through "
            "implications and edge cases. When asked to revise the spec, produce the complete "
            "revised spec between <revised_spec> and </revised_spec> tags, followed by a brief "
            "summary of the refinements you made."
        ),
        model=planner_model,
        permission_mode="bypassPermissions",
        allowed_tools=[],
        max_turns=2,
        effort=args.effort,
    )

    async with ClaudeSDKClient(options=planner_options) as planner:
        # Initialize planner with the spec
        async with Spinner("Initializing planner"):
            await planner.query(f"Here is the spec document you will be refining:\n\n{spec_text}")
            async for message in planner.receive_response():
                if isinstance(message, ResultMessage) and message.total_cost_usd:
                    total_cost += message.total_cost_usd

        current_spec = spec_text

        for i in range(1, args.iterations + 1):
            iter_start = time.monotonic()
            print_header(f"Iteration {i}/{args.iterations}")

            # Step 1: Questioner generates questions
            print_step(f"[1/3] Generating {args.questions} Socratic questions")
            async with Spinner("Questioner thinking") as sp:
                questions, q_cost = await run_questioner(
                    current_spec, args.questions, questioner_model, args.effort
                )
                total_cost += q_cost
            print(f"{DIM}     Done in {sp.elapsed:.0f}s{RESET}")
            print()
            for line in questions.strip().splitlines():
                print(f"  {YELLOW}{line}{RESET}")

            # Step 2: Planner answers questions — stream live
            print_step("[2/3] Planner answering questions")
            print()
            answer_start = time.monotonic()
            await planner.query(
                f"Here are {args.questions} probing questions about the current spec. "
                f"Think through each one carefully and answer in detail.\n\n{questions}"
            )
            answers_text = ""
            async for message in planner.receive_response():
                if isinstance(message, AssistantMessage):
                    text = extract_text(message)
                    answers_text += text
                    # Stream each line with indentation
                    for line in text.splitlines():
                        print(f"  {DIM}{line}{RESET}")
                elif isinstance(message, ResultMessage) and message.total_cost_usd:
                    total_cost += message.total_cost_usd
            answer_elapsed = time.monotonic() - answer_start
            print(f"\n{DIM}     Done in {answer_elapsed:.0f}s{RESET}")

            # Step 3: Planner revises spec — spinner (output is just the raw spec)
            print_step("[3/3] Revising spec")
            async with Spinner("Planner writing revised spec") as sp:
                await planner.query(
                    "Based on your answers above, revise the spec document to address the issues raised. "
                    "Output the complete revised spec between <revised_spec> and </revised_spec> tags, "
                    "followed by a brief summary of what refinements you made."
                )
                revision_text = ""
                async for message in planner.receive_response():
                    if isinstance(message, AssistantMessage):
                        revision_text += extract_text(message)
                    elif isinstance(message, ResultMessage) and message.total_cost_usd:
                        total_cost += message.total_cost_usd
            print(f"{DIM}     Done in {sp.elapsed:.0f}s{RESET}")

            # Extract revised spec
            revised = extract_revised_spec(revision_text)
            if revised:
                current_spec = revised
            else:
                print(f"{YELLOW}Warning: Could not extract <revised_spec> tags, using full response.{RESET}")
                current_spec = revision_text

            # Save version
            version_path = output_dir / f"v{i}.md"
            version_path.write_text(current_spec)

            # Print summary
            iter_elapsed = time.monotonic() - iter_start
            summary_match = re.search(r"</revised_spec>\s*(.*)", revision_text, re.DOTALL)
            summary = summary_match.group(1).strip() if summary_match else ""

            print(f"\n{GREEN}Saved {version_path}{RESET} {DIM}(iteration took {iter_elapsed:.0f}s){RESET}")
            if summary:
                print_step("Refinements made:")
                for line in summary.splitlines():
                    print(f"  {line}")
            print(f"{DIM}Running cost: ${total_cost:.4f}{RESET}")

    print_header("Complete")
    print(f"  Final spec: {GREEN}{output_dir / f'v{args.iterations}.md'}{RESET}")
    print(f"  Total cost: ${total_cost:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively refine a spec document using Socratic dialogue between two Claude agents."
    )
    parser.add_argument("spec_file", help="Path to the input spec document")
    parser.add_argument(
        "-q", "--questions", type=int, default=5, help="Number of Socratic questions per iteration (default: 5)"
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=3, help="Number of refinement iterations (default: 3)"
    )
    parser.add_argument("output_dir", help="Directory for versioned outputs")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model for both agents (default: claude-sonnet-4-5)")
    parser.add_argument("--questioner-model", default=None, help="Model for the Questioner agent (overrides --model)")
    parser.add_argument("--planner-model", default=None, help="Model for the Planner agent (overrides --model)")
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "max"], default=None,
        help="Reasoning effort level (default: high). 'max' is Opus 4.6 only.",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
