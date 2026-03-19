# plateau_restart

Tests the `plateau_recovery.cosine_restart` rule.

## Dynamics

- Steps 1-70: Normal decay for both `train_loss` and `val_loss`.
- Steps 80-120: `val_loss` flattens with range < 0.002, well within epsilon=0.003 over a window of 30 steps. `train_loss` continues a very slow drift.
- Steps 120+: After cosine restart fires (`lr_mult: 2.0`), both losses resume meaningful decay.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `plateau_recovery.cosine_restart` | `val_loss` range < 0.003 over 30 steps | `lr_mult: 2.0` (warm restart) |

## Usage

```bash
hotcb scenario run scenarios/plateau_restart/
```
