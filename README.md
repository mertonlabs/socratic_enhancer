# Socratic Enhancer

A CLI tool that iteratively refines any document by putting two Claude agents into a Socratic dialogue. A **Questioner** agent reads the document and asks probing questions. A **Planner** agent answers them, thinks through the implications, and rewrites the document. Repeat. Each pass surfaces gaps, resolves ambiguities, and adds detail you didn't know was missing.

## Why

Writing well is hard because you don't know what you don't know. The things that cause problems are rarely the things you thought about and decided — they're the things you never considered at all.

This tool automates the "poke holes in it" step. It's the equivalent of handing your document to a sharp colleague who asks uncomfortable questions, except it doesn't get tired and you can run it at 2am.

It works on anything that benefits from deeper thinking: technical specs, product briefs, prompts, proposals, research questions, architecture decision records, or even a rough paragraph of intent you want to flesh out. The Questioner reads the document, figures out what kind of document it is, and tailors its questions accordingly.

It's useful when:

- **You're about to build something non-trivial** and want to stress-test the plan before writing code.
- **You're crafting a prompt** and want to make sure it's precise enough to get the output you need.
- **You're writing alone** and don't have someone to review with. The Questioner acts as a skeptical reader.
- **Requirements are vague or inherited.** A few iterations will crystalize what's actually being asked for.
- **You want to go from rough idea to actionable document.** Start with a paragraph and let the loop do the thinking.

## How it works

```
For each iteration:

  1. Questioner (ephemeral)
     Reads the current document. Generates N questions targeting
     gaps, ambiguities, edge cases, and unstated assumptions.
     Question focus evolves: structural → behavioral → completeness.

  2. Planner (persistent)
     Receives the questions. Answers each one in detail,
     thinking through implications.

  3. Planner (continued)
     Revises the full document based on its own answers.
     Folds conclusions into existing text rather than appending.
     Saves the new version to disk.
```

The Questioner is stateless — it gets a fresh perspective each round. The Planner is persistent — it accumulates context across iterations, so later revisions build on earlier reasoning rather than starting over.

The key insight: the Planner's reasoning (in thinking tokens) gets distilled into the revised document. The document becomes a dense encoding of multiple rounds of deep thinking, compressed into something that fits in a context window. When you hand the final version to an implementation agent (or a person), they get the benefit of all that reasoning without needing to see the transcript.

Each version is saved as `v0.md` (original), `v1.md`, `v2.md`, etc., so you can diff between iterations or roll back.

## Install

Requires Python 3.11+ and the Claude Agent SDK:

```
pip install claude-agent-sdk
```

You'll need a valid Claude API key configured (via `ANTHROPIC_API_KEY` or Claude Code's auth).

## Usage

```
python socratic_enhancer.py <document> <output_dir> [options]
```

### Options

| Argument | Default | Description |
|----------|---------|-------------|
| `document` | (required) | Path to the input document |
| `output_dir` | (required) | Directory for versioned outputs |
| `-q`, `--questions` | `5` | Questions per iteration |
| `-i`, `--iterations` | `3` | Number of refinement passes |
| `--model` | `claude-sonnet-4-5` | Model for both agents |
| `--questioner-model` | (uses `--model`) | Override model for the Questioner |
| `--planner-model` | (uses `--model`) | Override model for the Planner |
| `--writer-model` | (uses `--planner-model`) | Model for writing revisions (see below) |
| `--effort` | `high` | Reasoning effort: `low`, `medium`, `high`, `max` (max is Opus 4.6 only) |

### Examples

Simplest run — 5 questions, 3 iterations, Sonnet for both agents:

```
python socratic_enhancer.py spec.md ./versions
```

Quick and cheap pass with fewer questions:

```
python socratic_enhancer.py brief.md ./versions -q 3 -i 2
```

Use a cheap model for questions, a powerful one for answering and revising:

```
python socratic_enhancer.py prompt.txt ./versions \
  --questioner-model claude-haiku-4-5 \
  --planner-model claude-opus-4-6
```

Opus for reasoning, Sonnet for writing — best of both:

```
python socratic_enhancer.py spec.md ./versions \
  --planner-model claude-opus-4-6 \
  --writer-model claude-sonnet-4-5 \
  --effort max
```

## Getting the most out of it

**Start with something, not nothing.** Even a rough paragraph is fine, but give it *something* to work with. A blank page or a single sentence won't produce useful questions. Include your intent, the key decisions you've already made, and any constraints you know about.

**More iterations > more questions.** 3 questions over 5 iterations tends to produce better results than 10 questions over 2 iterations. Each pass builds on the last, so you get compounding depth. Early rounds catch the obvious gaps; later rounds get into subtler territory.

**Diff the versions.** The real value is often in seeing *what changed* between iterations, not just the final output. Run `diff versions/v1.md versions/v2.md` (or use your editor's diff view) to see exactly what each round of questioning surfaced.

**Mix models deliberately.** The Questioner doesn't need to be expensive — it's just generating questions, and even smaller models ask good ones. The Planner benefits from capability since it needs to reason deeply. The Writer just needs to competently fold conclusions into text — it has the Planner's full reasoning in context, so it's a mechanical task. A good setup is Haiku for questions, Opus for reasoning, Sonnet for writing.

**Crank up effort for complex documents.** If your document involves tricky decisions or subtle trade-offs, `--effort max` (Opus 4.6 only) lets the Planner reason as deeply as it can. For Sonnet, `high` is the default and already the maximum.

**The output is a starting point.** The tool produces a more thorough document, not a perfect one. Read the final version critically. You'll often find that the tool surfaced the right questions but you'd answer some of them differently. That's the point — it's showing you what to think about.
