# Operator Quickstart

## Goal
This file tells the user exactly how to start a fresh remote Codex session so that it cleanly inherits the thesis rescue context.

## Workspace
Use:
- `/home/huangtu/PFL_clean_workspace/root_static`

## Preconditions
Before starting:
1. `codex` is installed and callable on the server
2. `OPENAI_API_KEY` is already exported
3. `~/.codex/config.toml` is configured correctly

## The Easiest Start
From the remote server shell:

```bash
cd /home/huangtu/PFL_clean_workspace/root_static
bash scripts/start_thesis_rescue_codex.sh
```

This will start Codex with the correct initial prompt and workspace.

## Manual Start If Needed
If the script fails for any reason:

```bash
cd /home/huangtu/PFL_clean_workspace/root_static
codex "$(cat docs/thesis_rescue/11_BOOTSTRAP_PROMPT.txt)"
```

## What The Fresh Session Should Do First
The fresh session should:
1. read `docs/thesis_rescue/00_MASTER_INDEX.md` through `09_REMOTE_STARTUP_PROMPT.md`
2. summarize:
   - the overall goal
   - locked historical facts
   - top 3 next actions
3. continue with stage3 static result consolidation

## What The Fresh Session Should Not Do First
- do not brainstorm a broad new defense
- do not jump back into PFedBA before static results are stabilized
- do not rely on hidden prior chat memory

## Most Important Directories
Mainline workspace:
- `/home/huangtu/PFL_clean_workspace/root_static`

Handoff docs:
- `/home/huangtu/PFL_clean_workspace/root_static/docs/thesis_rescue`

Key papers:
- `/home/huangtu/PFL_clean_workspace/root_static/docs/papers`

Current stage3 run:
- `/home/huangtu/PFL_clean_workspace/root_static/runs/20260417_184038_stage3_static_paper_common`
