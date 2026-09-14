# Book compatibility and corrections

These notebooks implement the algorithms taught in *Deep Reinforcement Learning
in Action*. Modern library APIs differ from the printed listings, and some
listings contain mistakes. The corrections below preserve the described methods;
they do not preserve erroneous operations verbatim. The architectures, genetic
operators, reward shaping, and principal training schedules remain those of the
chapter examples, except for the explicit Chapter 7 numeric errata below. Numerical results in the text are examples, not guarantees for
every random run.

## Environment compatibility

- Gymnasium returns `(observation, info)` from reset and separate `terminated` and
  `truncated` flags from step. Always reset after either end signal.
- Chapters 4 and 6 use CartPole-v1 with a 200-step limit to reproduce the book's
  CartPole-v0 horizon. Chapter 5 retains CartPole-v1's 500-step limit.
- ALE/Freeway-v5 is configured with RAM observations, random frame skipping
  `(2, 5)`, and sticky-action probability 0.25, matching the old Freeway-ram-v0
  registration. The chapter's three-action policy and reward shaping are retained.
- DoorKey's 400-step limit is passed to the actual environment constructor, not
  assigned to a wrapper. This is the finite-horizon task described by the book:
  its deadline ends the return, so terminal and deadline transitions are masked
  in the target. Mario's external progress cutoff instead restarts exploration;
  only true environment termination removes its bootstrap term.
- Playback uses RGB frames in the notebook, so a separate native display is not
  required. Close frames after displaying them. Battle records only its last
  training episode, rather than retaining all 100 episodes in memory.

## Corrections by chapter

### Chapter 3

`Gridworld.reward()` is available for readers copying printed listings. Blocked
moves leave the player in place and cost -1, as in the book. `makeMove()` also
returns `(reward, done)` for the maintained notebook. Chapter-local helper imports
are restored. Corner validation uses the final valid index `size - 1`. Restore the printed 64–150–100–4 network, mean squared
error and target synchronization every 500 action steps (listing 3.7), rather
than every 50 episodes. The randomized-grid examples explicitly use epsilon 0.3
from the original 2020 companion notebooks; the printed training loop omits this
reinitialization, while the PR changed it to 0.1. The 50-move exploration restart does not turn its last
state into an absorbing terminal state; actual pit and goal transitions do.

### Chapter 4

Restore the printed learning rate 0.0009 and the 200-step CartPole horizon.

### Chapter 5

The first Pool example imports `square` from a small module. Spawned processes
cannot import a function defined only in a notebook kernel; using an importable
worker is the portable form of the same multiprocessing example. No separate
Windows `multiprocess` package is needed. Worker scripts use the active Python
interpreter. Shared counters are locked and queue results are collected once per
worker rather than relying on `Queue.empty()`.

The n-step bootstrap uses the value of the state reached after the final step,
as required by the n-step return. Episode counts include the last transition.
The short return-calculation demonstration includes reward index zero.

### Chapter 6

The CartPole network returns log-probabilities. `Categorical(logits=...)` samples
the probabilities described in the text; passing them as `probs` reverses their
meaning. String mutation and selection settings are unchanged; exact recovery of
`Hello World!` is stochastic. In the supplementary MNIST notebook,
CrossEntropyLoss receives logits and individuals share a fitness-evaluation batch.
The supplementary standalone Agent uses its own environment and can save its
parameter tensors.

### Chapter 7

The chapter's distribution-update method is retained. Replay explicitly records
true termination and successful-crossing outcomes, so negative terminal rewards
cannot accidentally bootstrap. The stand-alone toy retains the printed
non-step-reward convention. Clear the parameter gradient before each backward
pass: successive minibatches must not silently accumulate earlier gradients.
Clamp probabilities before taking logarithms to avoid infinities from underflow.
Keep untaken actions anchored to the current-state prediction, as explained on
p.198; copying their next-state distributions instead changes actions for which
no transition target was observed.

Clearing accumulated gradients removes an unintended increase in effective update
size. To make the corrected implementation learn reliably in the tested runs,
three numeric settings are updated with the author's approval:

| Setting | Printed | Corrected |
|---|---:|---:|
| Simulated-data learning rate | 0.00001 | 0.005 |
| Freeway learning rate | 0.0001 | 0.01 |
| Freeway training updates | 1,300 | 10,000 |

The distributional algorithm, architecture, loss, replay capacity, batch size,
target synchronization, reward shaping and action policy are unchanged. The toy
still uses 1,000 updates. These are numeric errata, not an alternative method.

### Chapter 8

Softmax and epsilon-greedy operate over all 12 actions. Normalization uses the
action dimension, not the batch dimension. The inverse model returns logits to
CrossEntropyLoss, which supplies its own log-softmax. These implement the action
classification loss described by the text.

Each replay sample uses its own next-state maximum and terminal flag. The ICM
architecture, intrinsic-only reward, normalized Q loss, loss scales, optimizer,
and 5,000-update notebook budget are retained. Reset frame history between
episodes; include the actual terminal observation. Measure progress over the
configured window rather than comparing only the last action. The episode-length
plot now counts action decisions; horizontal distance has a separate plot.

### Chapter 9

Each sampled transition updates its own selected action, rather than the
cross-product of all batch actions. SGD updates the caller's parameter tensor
in place and clears its gradient. True agent deaths have no future-value term.
Stable softmax implements the same temperature policy without exponential
overflow. Full training records parameter changes and survivor counts; animation
alone cannot establish that learning occurred.

### Chapter 10

Positive rotation arguments actually rotate MNIST images. Single-image inference
retains its batch dimension; coordinate tensors follow the input device.

Double DQN selects the next-state action with the online model and evaluates that
action with the target model, as described in the book. The printed listing
accidentally used current-state action values for selection. Initialize the target
from the online network; keep the 100-update synchronization, 50% exploration,
positive-transition replay duplication, and 50,000-update training budget.
Record completed-episode lengths and successes, including timeouts.

### Appendix and historical notebook

Listing A.4 uses learning rate 0.001, 2,000 updates and batches of 100. These printed
settings are restored. Clear weight gradients per update, as intended by ordinary
gradient descent, and retain the two-layer network without substituting Adam.

The retained historical notebook uses modern reset/step APIs, explicit softmax
dimensions, detached plotting values and local output paths. Its actor-critic
uses the same one-step TD target for both the critic and advantage. The obsolete
Variable wrapper and incorrect demonstration loss call are corrected.

## Verification

Run `python -m unittest discover -s tests -v` for fast mathematical and API checks.
Execute each notebook from its chapter directory in a fresh kernel for full
training. The dependency pins are tested on Python 3.13; other Python versions and
GPU execution require separate validation. CPU execution is sufficient for the
examples. Fresh training and held-out evaluations are reported in
`VALIDATION.md`; do not infer learning from loss curves alone.
