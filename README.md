# WindyNav: Dual Prior + Adapter Alignment

This repo implements the experiment in `Instruction.md`:
- WindyNav environment with 4-band spatial wind fields (each band has linear time drift).
- Three datasets: D_action, D_state, D_align.
- VQ-VAE tokenizer for 64x64 frames.
- State prior + Action prior (decoder-only Transformers).
- Adapter that fuses state prior + action prior representations to predict actions.
- Baselines (scratch, action-only, state-only).
- Evaluation + visualization utilities.

## Setup

```bash
pip install -r requirements.txt
```

## Data generation

```bash
python scripts/generate_data.py --config configs/default.yaml --mode action --episodes 50000 --out data/D_action
python scripts/generate_data.py --config configs/default.yaml --mode state --episodes 50000 --out data/D_state
python scripts/generate_data.py --config configs/default.yaml --mode align --episodes 1000  --out data/D_align
```

## Train VQ-VAE tokenizer

```bash
python scripts/train_vqvae.py --data data/D_action --out models/vqvae.pt
```

## Tokenize frames

```bash
python scripts/tokenize_frames.py --data data/D_action --out tokens/D_action --vqvae models/vqvae.pt
python scripts/tokenize_frames.py --data data/D_state  --out tokens/D_state  --vqvae models/vqvae.pt
python scripts/tokenize_frames.py --data data/D_align  --out tokens/D_align  --vqvae models/vqvae.pt
```

## Train priors

```bash
python scripts/train_state_prior.py  --data tokens/D_state  --out models/state_prior.pt
python scripts/train_action_prior.py --data tokens/D_action --out models/action_prior.pt
```

## Train adapter (alignment)

```bash
python scripts/train_adapter.py \
  --data tokens/D_align \
  --state models/state_prior.pt \
  --action models/action_prior.pt \
  --out models/adapter.pt
```

## Baselines

```bash
python scripts/train_scratch_policy.py --data tokens/D_align --out models/scratch.pt
python scripts/train_state_only.py --data tokens/D_align --state models/state_prior.pt --out models/state_only.pt
```

## Evaluation

```bash
python scripts/eval_policy.py --model full --episodes 100 \
  --vqvae models/vqvae.pt --state models/state_prior.pt --action models/action_prior.pt --adapter models/adapter.pt \
  --out eval/full
```

Baselines:

```bash
python scripts/eval_policy.py --model scratch --episodes 100 \
  --vqvae models/vqvae.pt --scratch models/scratch.pt --out eval/scratch
python scripts/eval_policy.py --model action_only --episodes 100 \
  --vqvae models/vqvae.pt --action models/action_prior.pt --out eval/action_only
python scripts/eval_policy.py --model state_only --episodes 100 \
  --vqvae models/vqvae.pt --state models/state_prior.pt --state_only models/state_only.pt --out eval/state_only
```

## Visualization

```bash
python scripts/visualize.py --eval eval/full/eval_full.npz --out viz/full --num 20 --index 0
python scripts/visualize.py --eval eval/scratch/eval_scratch.npz --out viz/scratch --num 20 --index 0
python scripts/visualize.py --eval eval/action_only/eval_action_only.npz --out viz/action_only --num 20 --index 0
python scripts/visualize.py --eval eval/state_only/eval_state_only.npz --out viz/state_only --num 20 --index 0
```

## Success curve

Then plot:

```bash
python scripts/plot_success_curve.py \
  --full eval/full/eval_full.npz \
  --action_only eval/action_only/eval_action_only.npz \
  --state_only eval/state_only/eval_state_only.npz \
  --scratch eval/scratch/eval_scratch.npz \
  --out distance_curve.png \
  --success_radius 0.08
```

## Notes

- D_action uses wind=0 and straight-line tracking to teach line-following actions.
- D_state uses zero action with drifting wind.
- Adapter uses state summary + action-prior summary + goal to predict the action token.
- Training scripts save loss curves (`*_loss.png` and `*_loss.npy`) next to checkpoints.
