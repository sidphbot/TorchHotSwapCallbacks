# finish_checkpoint

Tests the `finish_strong.best_checkpoint` rule.

## Dynamics

- Steps 1-100: Near-converged training run. `val_loss` starts at 0.05 and decays exponentially.
- Around step ~60: `val_loss` drops below 0.01, triggering the `best_checkpoint` rule which enables the checkpoint callback.
- Steps 60+: Training continues with checkpointing active.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `finish_strong.best_checkpoint` | `val_loss < 0.01` | Enable checkpoint callback |

## Usage

```bash
hotcb scenario run scenarios/finish_checkpoint/
```
