# plateau_stagnation

Tests the `plateau_recovery.stagnation_detect` rule.

## Dynamics

- Steps 1-50: Normal decay from 2.0 toward ~1.0 with small noise.
- Steps 50-80: `train_loss` flattens with range < 0.001, well within the epsilon=0.002 threshold over a window of 25 steps. This triggers stagnation detection.
- Steps 80+: Slow recovery resumes (rule may have adjusted LR or other parameters).

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `plateau_recovery.stagnation_detect` | `train_loss` range < 0.002 over 25 steps | LR adjustment / plateau recovery |

## Usage

```bash
hotcb scenario run scenarios/plateau_stagnation/
```
