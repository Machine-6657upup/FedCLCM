# Project Goal And Pain Points

## User Situation
The user is under thesis deadline pressure. The project is not in a "nice to explore" phase. It is in a rescue phase.

That changes the objective:
- not maximum novelty
- not large method expansion
- not endless re-running to chase a memory
- instead: restore a stable, credible, writable experimental main line

## True Current Goal
The short-term thesis goal is:
1. establish a credible static-backdoor baseline
2. show FedRep is not invulnerable to static attacks
3. show static attacks are not trivially weak
4. compare FedAvg / FedRep / FedCLCM under one common regime
5. only then return to PFedBA and method modification

## Why This Became Urgent
Two severe contradictions created the current crisis.

### Pain Point 1: historical static results and recent static results diverged too much
Earlier experiments and the small paper narrative suggested:
- AdvPurge and FedRep could reach 85+ clean accuracy
- ResNet / ResNetP FedRep could be broken by BadNet / SIG / Blend

But many recent experiments suggested:
- clean accuracy around 78 or lower
- ASR around 5 or similarly low

That created the worst possible interpretation:
- low ASR may not mean defense is good
- low ASR may just mean the attack is weak

If that interpretation is true, the static section of the thesis collapses.

### Pain Point 2: root-level experiments and PFedBA-folder experiments do not tell the same story
In the root-level `main.py` experiment line, FedRep / FedCLCM sometimes looked resistant to:
- BadNet
- SIG
- Blend
- PFedBA

But in the PFedBA-specific folder and related runs:
- FedRep could be strongly broken
- FedCLCM also looked weak against PFedBA

So the problem was not one number being off. The whole experimental story became internally inconsistent.

## Why The Repository Situation Made This Worse
The user mainly developed in Cursor in a long-lived mixed workspace. As a result:
- the codebase became messy
- there were old and new scripts together
- dataset generation had historical modifications and hardcoded behavior
- logs were spread across multiple folders
- naming conventions drifted over time
- recent results were easy to misread or lose

So the rescue work had to do two things at once:
1. recover the truth from old logs
2. rebuild a clean forward path

## What "Success" Looks Like Now
Success does not mean exact reproduction of every old miracle number.

Success now means:
- a clean workspace
- known scripts
- known data generation parameters
- known attack strength
- known result directories
- known tables and curves
- a thesis argument that is consistent and defensible

## What Must Be Avoided
- Do not confuse low ASR with good defense unless clean accuracy is still healthy.
- Do not mix old logs and new logs without time and parameter tracing.
- Do not keep switching back to the historical mixed repository for new mainline experiments.
- Do not chase too many new ideas before static baselines are fully stabilized.
