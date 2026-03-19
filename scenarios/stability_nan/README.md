# stability_nan

Tests the `stability_basics.nan_guard` rule.

## Dynamics

- Steps 1-29: Normal loss decay from 2.0 with ~2% per-step reduction.
- Step 30: Emits `nan_detected=1` in the metrics dict, triggering the nan_guard rule.
- Step 31: Loss becomes NaN (simulating a real NaN event).
- Steps 32+: Loss resets and resumes decay at a reduced learning rate (halved by the rule's `lr_mult: 0.5` action).

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `stability_basics.nan_guard` | `nan_detected > 0` | `lr_mult: 0.5` |

## Usage

```bash
hotcb scenario run scenarios/stability_nan/
```
