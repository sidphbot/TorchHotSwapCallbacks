# multi_loss_dominance

## Policy Pack

`multi_loss_assist`

## Expected Rule

`multi_loss_assist.loss_ratio_target`

## Dynamics

- `train_loss` decays normally from 2.0 with smooth convergence.
- `aux_loss` tracks `train_loss` for the first ~50 steps, then grows to
  exceed 3x `train_loss`, triggering the `loss_ratio_target` rule.

## Expected Effect

The autopilot should decrease `aux_weight` to bring the loss ratio back
under the threshold.

## Running

```bash
python train.py  # or invoke via scenario runner
```
