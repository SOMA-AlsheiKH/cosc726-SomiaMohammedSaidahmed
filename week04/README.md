# Lab 4 — Layla as a CrewAI crew (VS Code)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

ollama pull qwen2.5:7b
ollama serve                       # in another terminal
```

In VS Code: **Ctrl/Cmd+Shift+P → Python: Select Interpreter → ./.venv**

## Run

```bash
python lab4_crewai.py gates        # start here: no model, costs nothing
python lab4_crewai.py prompt       # the prompt you never wrote
python lab4_crewai.py baseline     # the hand-rolled agent
python lab4_crewai.py crew         # the same job, as a crew
python lab4_crewai.py compare      # both, three fixtures
python lab4_crewai.py runaway      # max_iter=15 and what it costs
```

Five debug configurations are in `.vscode/launch.json` — use the Run and
Debug panel instead of the terminal if you prefer.

## Where to put your first breakpoint

`request_approval`, then run **gates**. It needs no model, so you can step
through all four checks for free and watch `SEEN` and `OBSERVED` change.

## The three things this lab is asking

1. **Where did the gates go?** They are in the tool bodies, because CrewAI
   gives you nowhere else. What does a new tool inherit?
2. **What is `SEEN`?** Module state. The Lab 3 dispatcher held it. What
   happens with two concurrent requests?
3. **Where is the pattern?** Run `prompt` and look at the generated schemas.
   `^A[0-9]{4}$` is not there. Type hints carry types, not judgements.
