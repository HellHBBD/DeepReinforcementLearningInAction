# Automated validation

`PR tests` runs on pull requests (the proposed merge commit), pushes to master,
merge-queue groups, and manual dispatch. Its **Regression and notebook smoke tests**
job runs 13 mathematical/API regression tests, five real-environment integration
tests, a runner failure-propagation test, and short unchanged sequences from all 13 notebooks. Smoke boundaries are
explicit in `scripts/notebooks.json`; a source fingerprint prevents cell shifts
from silently changing coverage. Smoke success does not imply convergence.

`Full training validation` runs weekly on Monday at 06:23 UTC and can be launched
from the Actions tab. It executes every nonempty code cell in all 13 notebooks,
with the notebook's configured training budgets and seed 46. Jobs run independently,
with at most three notebooks in parallel. The Chapter 6 job also executes the
standalone genetic CartPole program with its full default budget. Each notebook has a three-hour execution
limit. The job budget also allows time for dependency installation. Failures in one
notebook do not cancel the remaining notebooks.

The runner checks finite recorded losses/rewards, complete execution coverage,
Freeway's four-crossing criterion on five held-out environment seeds, nonzero
battle parameter changes, and at least 80% success in DoorKey's last 100 completed
training episodes. These are broad regression alarms. They do not replace the
multi-seed, held-out evaluations documented in `VALIDATION.md`, or establish exact
reproduction of every book figure. Gym randomness and multiprocessing scheduling
can still vary between runs.

Both workflows use Python 3.13 on Ubuntu 24.04 with the repository requirements
and CPU PyTorch wheels. They have read-only repository permissions, use ordinary
`pull_request` events, and need no secrets. Notebook execution happens in fresh
processes and temporary directories; source cells and training budgets are not
rewritten. Only animation delays/displays are suppressed. MNIST downloads are
shared locally under ignored `data/`. Logs, per-cell metrics, source hashes and
coverage summaries are uploaded even on failure (14 days for smoke tests, 30 days
for training). A failed shell worker also fails its notebook.

## Local commands

Install `requirements.txt`, then run from the repository root:

```sh
python -m unittest discover -s tests -v
python scripts/check_notebooks.py --mode smoke
python scripts/check_notebooks.py --mode full --notebook 'Chapter 7/Ch7_book.ipynb'
python scripts/check_notebooks.py --mode full
```

Results are written to `artifacts/notebooks/`; use `--output` to retain different
runs separately. Full runs download MNIST when needed and can take substantial
CPU time. Shell workers in Chapter 5 require OS multiprocessing/shared memory.

## GitHub activation and merge protection

The PR workflow will run after this commit is pushed (GitHub may require approval
for a fork's first workflow run). Scheduled and manual workflows become available
once their workflow files exist on the default branch. In repository branch
protection or a ruleset, require **Regression and notebook smoke tests** before
merging. Adding a workflow file does not itself enable that repository setting.
Do not require the weekly training matrix as a per-PR check: it has a separate
trigger and would leave PRs waiting for an unrelated run.

## Validation of this setup

Locally verified on Python 3.13: all 19 tests pass, including intentional Python,
shell-command and non-finite-loss failures; all 124 smoke cells across 13 notebooks
pass. Actionlint 1.7.7 accepts both workflows. The pinned PyTorch and torchvision
CPU wheels resolve for Python 3.13 on Linux. The full runner also passes Chapter 7
and its five-seed held-out crossing assertion, plus all 24 Chapter 5 cells and
both full multiprocessing training runs. GitHub-hosted execution still needs
the workflow commit to be pushed; local results do not certify the hosted runner.

GitHub's [workflow event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
explains PR merge-commit checks and default-branch requirements for scheduled and
manual runs.
