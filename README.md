# Socratic Enhancer

A CLI tool that iteratively refines a spec document by putting two Claude agents into a Socratic dialogue. A **Questioner** agent reads the spec and asks probing questions. A **Planner** agent answers them, thinks through the implications, and rewrites the spec. Repeat. Each pass surfaces gaps, resolves ambiguities, and adds detail you didn't know was missing.

## Why

Writing a good spec is hard because you don't know what you don't know. The things that sink a project are rarely the things you thought about and decided — they're the things you never considered at all.

This tool automates the "poke holes in the spec" step. It's the equivalent of handing your doc to a sharp colleague who asks uncomfortable questions, except it doesn't get tired and you can run it at 2am.

It's useful when:

- **You're about to build something non-trivial** and want to stress-test the plan before writing code. Cheaper to find a design flaw in a document than in a pull request.
- **You're writing alone** and don't have someone to review with. The Questioner acts as a skeptical reader.
- **Requirements are vague or inherited.** If someone gave you a loose brief, a few iterations will crystalize what's actually being asked for.
- **You want to go from "rough idea" to "buildable spec."** Start with a paragraph of intent and let the loop flesh it out.

It's probably overkill for specs that are already tight and well-scoped, or for changes small enough that the spec is just a sentence.

## How it works

```
For each iteration:

  1. Questioner (ephemeral)
     Reads the current spec. Generates N questions targeting
     gaps, ambiguities, edge cases, and unstated assumptions.

  2. Planner (persistent)
     Receives the questions. Answers each one in detail,
     thinking through implications.

  3. Planner (continued)
     Revises the full spec based on its own answers.
     Saves the new version to disk.
```

The Questioner is stateless — it gets a fresh perspective each round. The Planner is persistent — it accumulates context across iterations, so later revisions build on earlier reasoning rather than starting over.

Each version is saved as `v0.md` (original), `v1.md`, `v2.md`, etc., so you can diff between iterations or roll back.

## Install

Requires Python 3.11+ and the Claude Agent SDK:

```
pip install claude-agent-sdk
```

You'll need a valid Claude API key configured (via `ANTHROPIC_API_KEY` or Claude Code's auth).

## Usage

```
python socratic_enhancer.py <spec_file> [options]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-q`, `--questions` | `5` | Questions per iteration |
| `-i`, `--iterations` | `3` | Number of refinement passes |
| `-o`, `--output-dir` | `./versions` | Where versioned specs are saved |
| `--model` | `claude-sonnet-4-5` | Model for both agents |
| `--questioner-model` | (uses `--model`) | Override model for the Questioner |
| `--planner-model` | (uses `--model`) | Override model for the Planner |
| `--thinking-budget` | (none) | Max thinking tokens for extended thinking |

### Examples

Simplest run — 5 questions, 3 iterations, Sonnet for both agents:

```
python socratic_enhancer.py spec.md
```

Quick and cheap pass with fewer questions:

```
python socratic_enhancer.py spec.md -q 3 -i 2
```

Use a cheap model for questions, a powerful one for answering and revising:

```
python socratic_enhancer.py spec.md \
  --questioner-model claude-haiku-4-5 \
  --planner-model claude-opus-4-6
```

Deep reasoning with extended thinking:

```
python socratic_enhancer.py spec.md \
  --planner-model claude-opus-4-6 \
  --thinking-budget 10000
```

## Getting the most out of it

**Start with something, not nothing.** Even a rough paragraph is fine, but give it *something* to work with. A blank page or a single sentence like "build a todo app" won't produce useful questions. Include your intent, the key decisions you've already made, and any constraints you know about.

**More iterations > more questions.** 3 questions over 5 iterations tends to produce better specs than 10 questions over 2 iterations. Each pass builds on the last, so you get compounding depth. Early rounds catch the obvious gaps; later rounds get into subtler territory.

**Diff the versions.** The real value is often in seeing *what changed* between iterations, not just the final output. Run `diff versions/v1.md versions/v2.md` (or use your editor's diff view) to see exactly what each round of questioning surfaced.

**Mix models deliberately.** The Questioner doesn't need to be expensive — it's just generating questions, and even smaller models ask good ones. The Planner benefits more from capability since it needs to reason through answers and produce a coherent revised document. A common setup is Haiku for questions, Sonnet or Opus for planning.

**Use thinking budget for complex specs.** If your spec involves tricky architectural decisions or subtle trade-offs, giving the Planner a thinking budget (`--thinking-budget 10000`) lets it reason more carefully before answering. This matters most with Opus.

**The output is a starting point.** The tool produces a more thorough spec, not a perfect one. Read the final version critically. You'll often find that the tool surfaced the right questions but you'd answer some of them differently. That's the point — it's showing you what to think about.
