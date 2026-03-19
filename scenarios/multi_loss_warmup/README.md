# multi_loss_warmup

## Policy Pack

`multi_loss_assist`

## Expected Rule

`multi_loss_assist.aux_warmup_ramp`

## Dynamics

- Both `train_loss` and `aux_loss` decay slowly from high initial values.
- `aux_weight` starts at 1.0 (full weight from the beginning).
- The run stays below step 100 for most of its duration, which triggers
  the `aux_warmup_ramp` rule (step < 100 condition).

## Expected Effect

The autopilot should ramp `aux_weight` gradually during the warmup phase
rather than applying full auxiliary loss weight from the start.

## Running

```bash
python train.py  # or invoke via scenario runner
```
