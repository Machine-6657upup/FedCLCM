# Remote Operating Rules

## Main Working Principle
The remote Codex session should behave like a careful thesis rescue assistant, not like a brainstorming partner.

That means:
- preserve momentum
- narrow scope
- verify through logs and files
- write down results

## Trust Hierarchy
Trust order:
1. repository docs in `docs/thesis_rescue/`
2. actual code and scripts
3. actual logs and run directories
4. prior memory only when it is explicitly written down

## Workspace Rule
Mainline workspace:
- `/home/huangtu/PFL_clean_workspace/root_static`

Historical workspace:
- `/home/huangtu/PFL_Backdoor_Defense`

Rule:
- run and edit mainline experiments in the clean workspace
- use the historical workspace only for comparison, old logs, and old-code tracing

## Logging And Run Discipline
Every new run should have:
- a dedicated run directory
- clear script provenance
- explicit key hyperparameters
- dataset fingerprint or attack-generation fingerprint when applicable
- final summary written into docs

Do not allow new important results to exist only in terminal memory.

## Result Interpretation Rule
Every claimed defense result must answer:
1. what is the clean accuracy
2. what is the ASR
3. does it converge
4. does it collapse
5. does it support a coherent thesis claim

## Startup Prompt Recommendation
When a fresh remote Codex starts, the user should give a prompt like:

"First read `docs/thesis_rescue/00_MASTER_INDEX.md` through `08_REMOTE_OPERATING_RULES.md`. Then summarize:
1. the overall thesis rescue goal
2. the locked historical facts
3. the top 3 next actions
Do not propose new large methods before the static line is fully stabilized."

## Papers To Keep Available On Server
Recommended server path:
- `/home/huangtu/PFL_clean_workspace/root_static/docs/papers`

Recommended files:
- `BDPFL`
- `PFL-ALP`
- `Bad-PFL`
- `FLIGHT`
- `usenixsecurity24-lyu.pdf`
- `AdvPurge.pdf`
- `main.pdf`

## Why These Rules Exist
The entire rescue process became difficult because too much useful information lived only in transient chat memory or scattered logs.

The remote workflow should explicitly prevent that from happening again.
