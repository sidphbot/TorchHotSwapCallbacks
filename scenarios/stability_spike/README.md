# stability_spike

Tests the `stability_basics.grad_spike_clip` rule.

## Dynamics

- Steps 1-19: Normal training with `grad_norm` between 1 and 3.
- Steps 20-25: `grad_norm` spikes to 15-20, well above the threshold of 10. Loss destabilizes slightly.
- Steps 26+: `grad_norm` returns to normal range (1-3), loss resumes decay.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `stability_basics.grad_spike_clip` | `grad_norm > 10` | Gradient clipping / LR adjustment |

## Usage

```bash
hotcb scenario run scenarios/stability_spike/
```
