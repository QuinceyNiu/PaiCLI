# PaiCli

PaiCli is a Python CLI agent project focused on practical coding-assistant workflows. It provides an interactive terminal experience with ReAct tool use, plan-and-execute task handling, memory, code retrieval, multi-agent collaboration, and human-in-the-loop approval.

## Features

- **ReAct Agent**: reasons through tasks, calls tools, observes results, and continues until it can answer.
- **Plan-and-Execute**: turns complex goals into executable plans, shows the plan, and asks for confirmation before running it.
- **Memory System**: keeps short-term conversation context, extracts long-term facts, tracks token usage, and compresses context when needed.
- **RAG for Codebases**: indexes source files, chunks code, builds lightweight vector search, and supports semantic code search.
- **Code Graph Queries**: stores simple code relations and lets you inspect class or method relationships.
- **Multi-Agent Mode**: coordinates planner, executor, and reviewer-style roles for larger tasks.
- **HITL Approval**: wraps tools with human approval policies so sensitive operations can require confirmation.
- **CLI Tooling**: includes file, directory, shell-command, project-creation, and code-search tools.

## Project Structure

```text
.
├── paicli/
│   ├── agent/      # ReAct, Plan-and-Execute, and Multi-Agent orchestration
│   ├── cli/        # Interactive command line entrypoint
│   ├── hitl/       # Human-in-the-loop requests, policies, and terminal handler
│   ├── llm/        # GLM API client
│   ├── memory/     # Conversation memory, long-term memory, and token budgeting
│   ├── plan/       # Task and execution-plan models
│   ├── rag/        # Code indexing, chunking, embedding, retrieval, and formatting
│   └── tool/       # Tool registry and HITL-aware tool registry
├── tests/          # Unit tests
├── pyproject.toml  # Package metadata and console script
└── .env.example    # Example environment configuration
```

## Requirements

- Python 3.11+
- A GLM API key

## Installation

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Configuration

Copy the example environment file and fill in your API key:

```bash
cp .env.example .env
```

Required variable:

```bash
GLM_API_KEY=your_api_key_here
```

Optional variables such as `GLM_BASE_URL` and `GLM_MODEL` can also be configured in `.env`.

## Usage

Run the CLI after installing the package:

```bash
paicli
```

Or run it directly as a module:

```bash
python -m paicli.cli.main
```

Common interactive commands:

```text
/plan                 Enter Plan-and-Execute mode for the next complex task
/team                 Run the next task through the Multi-Agent workflow
/hitl [on|off]        View or toggle human-in-the-loop approval
/memory               Show memory and token status
/index [path]         Index a codebase for RAG search
/search <query>       Search the indexed codebase
/graph <name>         Show stored relations for a class or method
/save <fact>          Save a fact into long-term memory
/clear                Clear conversation history after extracting key facts
exit or quit          Leave the CLI
```

## Development

Run the test suite from the project root:

```bash
python -m unittest discover
```

Useful Git workflow for future module development:

```bash
git status
git add paicli tests README.md pyproject.toml .gitignore .gitattributes .env.example
git commit -m "chore: prepare python project structure"
```

## Notes

- `.env`, virtual environments, caches, build outputs, IDE metadata, and generated package metadata are ignored by Git.
- Keep new feature work inside focused modules under `paicli/`, with matching tests under `tests/`.
