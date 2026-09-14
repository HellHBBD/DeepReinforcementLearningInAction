# Validation of the book-compatible corrections

**Implementation fixes include the author-approved Chapter 7 numeric errata. Full training and outcome checks are recorded below; illustrative book results are not guaranteed for every run.**

Base: PR #46, `66862fbce7b3cf073cf7e64ce61f96f5f89efe6e`.
The supplied published PDF takes precedence over newer repository changes. Where
printed listings omit setup, the author's 2020 companion checkout is the reference.
See [BOOK_COMPATIBILITY.md](BOOK_COMPATIBILITY.md) for the exact corrections.

## Method

CPU, Python 3.13.2, the complete corrected requirements. A clean resolver dry run
(`pip install --dry-run --ignore-installed -r requirements.txt`) passes. This is
not a claim of validation on Windows, CUDA, or Python 3.14.

All 276 code cells in 13 notebooks execute without errors in isolated IPython namespaces, with full configured
training. Matplotlib is headless; display delays are omitted by the audit harness.
Chapter 5 runs with shared-memory access outside the filesystem sandbox. Its
previous interactive-pool failure is fixed, not skipped. Notebook outputs are
cleared in the source files; execution logs, metrics and checkpoints are retained
in the local review artifacts. Random seeds are identified below where repeated;
Gym reset randomness and parallel scheduling can introduce additional variation.

Thirteen fast checks cover mathematical/API contracts, including actual parameter
mutation, per-sample targets, action-axis softmax, current-state targets for untaken
actions, next-state Double DQN selection, rotation, finite horizons and checkpoint
saving. Run them with `python -m unittest discover -s tests -v`.

## Measured outcomes

| Example | Full-run observation |
|---|---|
| Chapter 1 | All cells execute; Fibonacci results and memoization agree. |
| Chapter 2 | All cells and 10,000 contextual-bandit updates execute. |
| Chapter 3 | Three full repeats (seeds 46–48), each with 5,000 replay episodes and 5,000 target-network episodes: replay wins 86.3%, 87.5%, 82.7%; target wins 86.1%, 90.3%, 87.3%, evaluated over 1,000 games each. These are below the book’s illustrative ~90%/~95%. |
| Chapter 4 | Printed learning rate and 200-step horizon; 100 evaluation episodes average 119.12 steps after 500 training episodes. |
| Chapter 5 | All cells execute, including the Pool example and both multiprocessing runs. Final n-step policy: stochastic mean 373.65 steps, greedy 416.91 over 100 held-out episodes each. |
| Chapter 6 | Full 20-generation CartPole run; demonstration reaches 200 steps. String genetic operators and settings are unchanged. |
| Supplementary genetic MNIST | Full 50 genetic generations and 50 Adam epochs execute. Held-out accuracy: genetic 13.74%, Adam 91.35%. The simple evolutionary baseline remains weak; no specific book accuracy is claimed for this supplementary notebook. |
| Chapter 6 standalone | Full 50 generations × 100 agents × 10 trials; final population average 94.71, best final fitness 136.6, demonstration 87 steps. |
| Chapter 7 | Adopted numeric errata: toy learning rate 0.005, Freeway learning rate 0.01 and 10,000 updates. Three training seeds (0, 1, 46), each evaluated on five environment seeds, achieve 11, 11, 11, 11, 10 crossings in 1,300 steps. Seeds 1 and 46 start with zero crossings. All three toy runs place their modes at the correct target values 0 and 8, with approximately 96–98% mass within one neighboring support atom. |
| Chapter 8 | Full 5,000-update run and all notebook cells complete. Five held-out evaluations: trained maximum x positions 1,416, 1,258, 1,412, 1,138, 808; mean 1,206.4, versus means 46.0 for the actual untrained checkpoint and 421.2 for random actions. This demonstrates forward progress and jumping; none completes the level. |
| Chapter 9 | Full 100 episodes complete with no errors. Parameter-change norms are 2.15 and 2.32 for the two teams. Three paired evaluation seeds: mean combined team return improves from -739.73 to -564.83; successful attacks occur, but no agents die in the trained evaluation episodes. This supports improved reward, not a claim of a stronger killing strategy. |
| Chapter 10 MNIST | 1,000 updates with actual rotation; 94.33% over the complete 10,000-image perturbed test set, consistent with the book's nearly 95%. |
| Chapter 10 DoorKey | Full 50,000 updates complete. 200 held-out episodes per policy: book’s epsilon=0.5 policy 100% success, greedy 90%, random 48%, all with an actual 400-step cap. Zero steps occur after termination/truncation. The book’s ≥94% is reproduced with its stated exploration policy, not with greedy evaluation. |
| Appendix | Printed 2,000 updates, batch 100, learning rate 0.001, ordinary gradient descent: 65.79%, 67.46%, 67.28% across seeds 46–48 on all 10,000 test images. The book's roughly 70% is illustrative, not an exact reproduced number. |
| Historical policy-gradient notebook | All cells execute under current dependencies; both 500- and 750-episode training sections complete. |

The final working-copy Chapter 7 notebook was rerun in full after applying the
errata. Its trained policy again achieved 11, 11, 11, 11, 10 crossings, versus
zero for its initial policy and random actions. This matches the always-up
baseline on these seeds; it establishes the book’s crossing behavior, not a
strategy superior to that simple baseline. The final toy places 98.08% and 96.38%
of probability near the respective target values.

## Constraints on claims

Correcting errors in a printed listing necessarily changes those operations. The
intent is fidelity to the described algorithm, not preservation of errors. In
particular, accumulated gradients made the effective updates in Chapter 7 much
larger than its nominal learning rate. Correcting that error does not guarantee
that the same small rate and short budget will reproduce the published plot.

The original Chapter 7 rates and 1,300-step Freeway budget failed to learn from
initially unsuccessful seeds after gradient clearing. The toy loss barely changed
(23.44 → 23.37 for the first/last 100 updates). The three explicit numeric errata
in BOOK_COMPATIBILITY.md address this with the author's approval. Architecture,
distribution updates, loss, replay and policy remain the book's method. The
successful seed-0 untrained policy is reported as such; seeds 1 and 46 provide
evidence of acquisition. Finite multi-seed tests do not guarantee convergence for
every random seed.

The three repeated full Chapter 3 runs, three Appendix runs, and paired policy
evaluations are additional to this complete notebook pass. The standalone genetic
program also completes its full default 50-generation run; a separate CLI check
verifies integer options and checkpoint saving. Printed chapter-local helper
imports and consistency of the Chapter 5 generated helper were checked.
