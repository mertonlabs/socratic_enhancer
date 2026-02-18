#!/usr/bin/env python3
"""Socratic Enhancer: Iteratively refine a document using two Claude agents in a Socratic dialogue loop."""

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


QUESTIONER_FOCUS = {
    "early": (
        "This is an early iteration. Focus on structure and intent:\n"
        "- Is the purpose and scope clearly defined?\n"
        "- Are the major components, concepts, or sections identified and justified?\n"
        "- Are there key decisions or trade-offs that haven't been explicitly made?\n"
        "- Are there dependencies, constraints, or assumptions that aren't stated?\n"
        "- What would someone acting on this document need to decide for themselves?"
    ),
    "mid": (
        "This is a mid iteration. Focus on interactions, edge cases, and consistency:\n"
        "- Where do different parts of the document interact — are those boundaries clear?\n"
        "- What are the failure modes or scenarios where things go wrong?\n"
        "- Are there contradictions, ambiguities, or places where the document says two different things?\n"
        "- What happens in unusual or extreme cases?\n"
        "- What would someone misunderstand or get wrong when reading this?"
    ),
    "late": (
        "This is a late iteration. Focus on completeness and actionability:\n"
        "- Could someone act on this document without making any assumptions?\n"
        "- Are all details, constraints, and criteria fully specified?\n"
        "- What practical concerns are missing (resources, risks, sequencing)?\n"
        "- Is anything vague where it should be precise, or over-specified where it should be flexible?\n"
        "- What would go wrong in practice that this document doesn't address?"
    ),
}


def get_questioner_focus(iteration: int, total_iterations: int) -> str:
    """Return the appropriate question focus based on iteration progress."""
    if total_iterations <= 1:
        return QUESTIONER_FOCUS["early"]
    progress = (iteration - 1) / (total_iterations - 1)
    if progress < 0.34:
        return QUESTIONER_FOCUS["early"]
    elif progress < 0.67:
        return QUESTIONER_FOCUS["mid"]
    else:
        return QUESTIONER_FOCUS["late"]


async def run_questioner(
    spec_text: str, num_questions: int, model: str, effort: str | None,
    iteration: int, total_iterations: int, focus: str | None = None,
) -> tuple[str, float]:
    """Run an ephemeral Questioner agent. Returns (questions_text, cost)."""
    iteration_focus = get_questioner_focus(iteration, total_iterations)
    user_focus = f"\nFocus your questions specifically on: {focus}\n" if focus else ""
    prompt = f"""Read the following document. First, understand what kind of document it is and what it's trying to achieve. Then generate exactly {num_questions} probing questions tailored to the document's purpose.

{iteration_focus}
{user_focus}
Ask questions that force concrete, specific answers — not questions that can be answered with "yes" or "it depends." Each question should target something that, if left unresolved, would cause someone acting on this document to guess or make an assumption.

Output each question on its own line, numbered (1., 2., etc.).

<document>
{spec_text}
</document>"""

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a critical analyst. Read the document, infer what kind of document it is "
            "(e.g. technical spec, product brief, prompt, proposal, plan, research question, etc.), "
            "and ask questions appropriate to its type and purpose. Your questions should surface "
            "gaps, ambiguities, and unstated assumptions that would matter to whoever acts on this "
            "document. Ask only questions — do not answer them or suggest solutions."
        ),
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
    questioner_model = args.questioner_model or args.model

    if args.questions_only:
        print_header("Socratic Enhancer — Questions Only")
        print(f"  Document:   {spec_path}")
        print(f"  Questions:  {args.questions}")
        print(f"  Questioner: {questioner_model}")
        if args.focus:
            print(f"  Focus:      {args.focus}")
        if args.effort:
            print(f"  Effort:     {args.effort}")
        async with Spinner("Questioner thinking") as sp:
            questions, cost = await run_questioner(
                spec_text, args.questions, questioner_model, args.effort, 1, 1,
                focus=args.focus,
            )
        print(f"{DIM}     Done in {sp.elapsed:.0f}s{RESET}\n")
        for line in questions.strip().splitlines():
            print(f"  {YELLOW}{line}{RESET}")
        print(f"\n{DIM}Cost: ${cost:.4f}{RESET}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save v0
    v0_path = output_dir / "v0.md"
    v0_path.write_text(spec_text)

    planner_model = args.planner_model or args.model
    writer_model = args.writer_model or planner_model

    print_header("Socratic Enhancer")
    print(f"  Document:   {spec_path}")
    print(f"  Questions:  {args.questions}/iteration")
    print(f"  Iterations: {args.iterations}")
    print(f"  Questioner: {questioner_model}")
    print(f"  Planner:    {planner_model}")
    if writer_model != planner_model:
        print(f"  Writer:     {writer_model}")
    if args.focus:
        print(f"  Focus:      {args.focus}")
    if args.effort:
        print(f"  Effort:     {args.effort}")
    print(f"  Output:     {output_dir}/")
    print(f"\n{DIM}Saved initial spec as v0.md{RESET}")

    total_cost = 0.0

    planner_options = ClaudeAgentOptions(
        system_prompt=(
            "You are a document refinement agent. You will be given a document and asked "
            "probing questions about it. Answer each question thoroughly, thinking through "
            "implications and edge cases. When asked to revise the document, produce the complete "
            "revised document between <revised_spec> and </revised_spec> tags, followed by a brief "
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
            await planner.query(f"Here is the document you will be refining:\n\n{spec_text}")
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
                    current_spec, args.questions, questioner_model, args.effort,
                    i, args.iterations, focus=args.focus,
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
                f"Here are {args.questions} probing questions about the current document. "
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

            # Step 3: Revise document — switch to writer model if different
            print_step("[3/3] Revising document")
            if writer_model != planner_model:
                await planner.set_model(writer_model)
            async with Spinner("Writing revised document") as sp:
                await planner.query(
                    "Based on your answers above, revise the document. Fold your conclusions "
                    "directly into the existing text — strengthen and clarify what's already there rather "
                    "than appending new sections. The goal is a denser, more precise document, "
                    "not a longer one. Every sentence should reduce ambiguity for whoever acts on this.\n\n"
                    "Output the complete revised document between <revised_spec> and </revised_spec> tags, "
                    "followed by a brief summary of what refinements you made."
                )
                revision_text = ""
                async for message in planner.receive_response():
                    if isinstance(message, AssistantMessage):
                        revision_text += extract_text(message)
                    elif isinstance(message, ResultMessage) and message.total_cost_usd:
                        total_cost += message.total_cost_usd
            if writer_model != planner_model:
                await planner.set_model(planner_model)
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
        description="Iteratively refine a document using Socratic dialogue between two Claude agents."
    )
    parser.add_argument("spec_file", metavar="document", help="Path to the input document")
    parser.add_argument(
        "-q", "--questions", type=int, default=5, help="Number of Socratic questions per iteration (default: 5)"
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=3, help="Number of refinement iterations (default: 3)"
    )
    parser.add_argument(
        "--questions-only", action="store_true",
        help="Generate and print one set of questions without running the planner or writer"
    )
    parser.add_argument(
        "--focus", default=None,
        help="Optional focus area to direct the questioner (e.g. 'security', 'error handling')"
    )
    parser.add_argument("output_dir", nargs="?", default=None, help="Directory for versioned outputs")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model for both agents (default: claude-sonnet-4-5)")
    parser.add_argument("--questioner-model", default=None, help="Model for the Questioner agent (overrides --model)")
    parser.add_argument("--planner-model", default=None, help="Model for the Planner agent (overrides --model)")
    parser.add_argument("--writer-model", default=None, help="Model for writing revisions (defaults to planner model)")
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "max"], default=None,
        help="Reasoning effort level (default: high). 'max' is Opus 4.6 only.",
    )

    args = parser.parse_args()
    if not args.questions_only and args.output_dir is None:
        parser.error("output_dir is required unless --questions-only is set")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
