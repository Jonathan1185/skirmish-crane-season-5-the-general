# Skirmish at Crane Reach: Imitate Your Army

In this example, a game AI agent you already have becomes the **teacher**, and a new agent, the **learner**, learns to imitate it. You record the teacher's choices for footmen, archers, and cavalry, train a small neural network on them with [PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html), and compare the two armies on new seeds. This approach is called **imitation learning**.

This branch is a complete, runnable agent repository. It ships with a stand-in teacher, the **Banner** army from Season 4, and a learner already trained from it, so every command below works before you change anything. Start with the [Getting Started guide](https://vox-deorum.github.io/game-sandbox/students/getting-started/) if you have not set up a project before. Its [Optional: use PyTorch](https://vox-deorum.github.io/game-sandbox/students/getting-started/#optional-use-pytorch) section explains how PyTorch runs on the course server.

## Set up

Create `.venv` and install the requirements, which already include PyTorch:

```console
python -m sandbox setup
```

Activate `.venv`. In PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

If you cannot activate `.venv`, replace `python` with `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS or Linux. Then run the checks:

```console
python -m sandbox test
```

Run every command from this folder.

### Bring your own teacher

The teacher is `teacher.py`, and Banner's helper `blocks.py` comes with it. To learn from your own agent instead:

1. Copy your agent's `agent.py` into this folder as `teacher.py`, replacing Banner's.
2. Copy its helper files and saved models too, replacing `blocks.py` if your agent has one. If another helper has the same name as a learner file, such as `encoder.py`, `routes.py`, or `imitation-policy.pt`, rename it and update the code that imports it.
3. `season.json` holds the Season 4 settings Banner was built for. To use your own season's settings, follow **Set Up Locally** on **My Submissions**, which replaces it. The local [`environment.md`](environment.md) guide covers the actions, observations, and limits.
4. Collect, train, and evaluate again, as described next.

The teacher needs a constructor without arguments and the usual `reset(seed, observation)` and `act(observation)` methods. If its class in `teacher.py` has a name other than `Agent`, pass `--teacher teacher:MyAgent` to collection and evaluation.

### What is in this folder

- **The learner**, at the root: `agent.py`, `encoder.py`, `imitation_model.py`, and the trained model `imitation-policy.pt`. `routes.py` is a helper for the route-cost exercise in **Practice**; nothing imports it until you add that feature. `manifest.json` points at `agent.py`, so this folder is what you submit.
- **The teacher**: `teacher.py` and its helpers. The learner never imports them.
- **The experiment**, in `imitation/`: the scripts that collect, train, and evaluate. The learner never imports them either.
- **The results**, in `runs/`: the scripts create this folder, and `.gitignore` keeps it out of your repository.

## Run the experiment

Collect demonstrations, train the learner, compare both agents on new seeds, then test and watch the learner:

```console
python -m imitation.experiment collect --episodes 10
python -m imitation.train runs/demonstrations.jsonl
python -m imitation.experiment evaluate --start 100 --episodes 3
python -m sandbox test
python -m sandbox watch --seed 100
```

**Collect.** For each of ten seeds, the teacher plays one match from each seat against Naive. Every unit activation adds one **demonstration** to `runs/demonstrations.jsonl`. Check the printed counts for all three unit types. If the teacher returns an invalid action or times out, collection stops without saving, so fix the problem and collect again. The `demonstration` function in `imitation/experiment.py` builds each line:

| Field | Contents |
| --- | --- |
| `entities` | One row of numbers for the acting unit, each visible ally or enemy, and each capture zone |
| `move_options` | One row per reachable tile |
| `target_options` | One row per possible target, measured from the teacher's tile |
| `target_entity_indices` | Which enemy row each target names; `-1` means no target |
| `move_choice`, `target_choice` | The teacher's chosen rows |

**Train.** To check the learner on matches it never trained on, training holds out seeds 8 and 9 for validation. It stops after 15 epochs (passes through the data) without improving the validation loss, or at the `--epochs` limit of 200, and keeps the best model. It writes `imitation-policy.pt` and a report to `runs/imitation-policy.training.json`. `watch`, `eval`, and the tests all load `imitation-policy.pt`, so after training they play your new learner.

`--width`, `--layers`, and `--heads` set the model's size. `--epochs`, `--batch-size`, `--threads`, `--device`, and `--seed` control the run. Training uses a GPU when PyTorch finds one (CUDA on an NVIDIA card, or MPS on a Mac), and the learner always plays on the CPU. For example:

```console
python -m imitation.train runs/demonstrations.jsonl --batch-size 32 --device cpu
```

**Evaluate.** Evaluation writes `runs/evaluation.json` for twelve matches: two agents, two seats, and three seeds. It records each army's scores, `act` times, and timeouts.

Collection and evaluation refuse to overwrite a file, so give each new run its own `--output`. Training replaces its output, so pass `--output runs/revised.pt` to keep a revision. To play a revision, copy it to `imitation-policy.pt`, the file the learner loads and you submit.

## Read the code

Each core file holds one idea. We recommend opening them in this order:

| File | Idea | Start with |
| --- | --- | --- |
| `imitation/experiment.py` | The teacher's actions become demonstrations | `demonstration` |
| `agent.py` | The output: encode once, score tiles, then targets | `choose_action` |
| `encoder.py` | What the learner sees | `ENTITY_FEATURES`, `encode_entities`, then `describe_move` |
| `imitation_model.py` | How attention scores the options | `Policy.forward`, `score_moves`, `score_targets` |
| `imitation/train.py` | The training loop and its report | `fit_model`, then `main` |

To make sure collection and play read the same features, datasets and models record the encoder's `VERSION`, and training and the learner check that it matches. Give `VERSION` a new value whenever you change a feature.

You can follow the experiment without the supporting files. `imitation/experiment_tools.py` runs matches and gives each unit its own teacher, and `imitation/train_tools.py` batches demonstrations and summarizes the report.

## Practice

Each exercise asks one design question. You can make the change yourself or describe it to a coding agent, then ask it which files it changed and why. To compare fairly, train each revision to its own `--output`, keep the validation seeds, and evaluate on the same seeds. If a change alters a feature, change `VERSION` and collect again with a new `--output`.

1. **Find what the learner misses.** Open the `disagreements` in the training report. Each one names a seed, seat, and unit, with the teacher's tile and the learner's (`[1, 0]` means one step east). Watch that seed and seat, for example `python -m sandbox watch --seed 8 --seat 1`. What does the teacher consider that the features leave out?
2. **Change what the model sees.** Add a route cost to each enemy's row with `route_cost` from `routes.py`, or remove a feature such as `weakest_enemy`. How do tile agreement by unit type and `act` time change?
3. **Change what the model sends out.** Have the learner choose its target before its tile, or score each tile and target pair together. Compare tile, target, and whole-action agreement, then watch how cavalry strike.
4. **Change the model's size.** Train with `--width 32 --layers 1`, the defaults (`--width 64 --layers 2`), and `--width 128 --layers 3`, keeping `--heads 4`. Compare parameters, training time, validation loss, agreement, and `act` time.
5. **Change what the model learns.** Count a missed retreat more heavily than a missed step forward, or give credit for any tile as safe as the teacher's. Does agreement fall while the archers play better?
6. **Correct the learner.** Save your first model and report as `runs/first.pt` and `runs/first.training.json`. Then let the learner play while the teacher supplies corrections, and retrain on both files:

   ```console
   python -m imitation.experiment collect --driver learner --output runs/corrections.jsonl
   python -m imitation.train runs/demonstrations.jsonl runs/corrections.jsonl
   python -m imitation.experiment evaluate --start 100 --episodes 3 --output runs/corrected.json
   ```

   Compare `teacher_driven` and `learner_driven` agreement in the new report, then compare both learners on new seeds.
7. **Change the teacher.** In a second copy of this folder, train a learner from an earlier version of your agent. Then play the two learners against each other from both seats:

   ```console
   python -m sandbox eval --seeds 200 201 202 --seat 0 --vs ../second-folder
   python -m sandbox eval --seeds 200 201 202 --seat 1 --vs ../second-folder
   ```

   Which of its teacher's choices did each learner keep?

## Read the results

- **Agreement.** Compare each unit type with `random_choice_baseline` in the report. The shipped learner, trained from Banner on seeds 0 to 9, picks Banner's tile 86.9% of the time on the validation seeds, against 9.6% for a random legal choice. Cavalry reach more tiles than footmen and agree less often. Target agreement is usually high because most decisions offer one or two targets. The report measures targets from the teacher's tile, so use evaluation to see the learner's own choices.
- **Time.** Encoding the battlefield usually takes longer than running the network. Watch `act` time after adding a feature that runs a search, such as a route cost. The first `act` call of a match can be slower, so evaluation reports it separately.
- **Seeds.** Keep a set of new seeds for the final comparison. If an evaluation shapes a revision, choose different seeds for the next one.

## Submitting

The learner submits like any other agent: follow the shared [submitting guide](https://vox-deorum.github.io/game-sandbox/students/submitting/). The course server provides its own copy of PyTorch, and `imitation-policy.pt` counts toward the season's size limit, so keep the model small. The server never runs the scripts in `imitation/`.

##Design Goal: My design goal was to make the archer smarter by helping it try to get away from the enemy and make it go close to the footmen. In addition to this the calvalry go into the capture area and once they perish the footmen go then the archers. Additionally I tried making the footmen more focused on protecting the archer while the calvary is still alive.Additionally, the different npcs should try to capture the closest capture zone.

##Short Reflection: I also wanted the capture area to play a major part of it so I made the calvary job to capture it then the footmen then the archers. Additionally the different roles try to capture a different capture zone.

#AI Disclosure and reflection: I used a AI agent (Codex) for my agent to help with the unit behavior and agent navigation.