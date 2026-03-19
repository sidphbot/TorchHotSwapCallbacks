# stability_divergence

Tests the `stability_basics.loss_spike_recovery` rule.

## Dynamics

- Steps 1-35: Normal loss decay from 2.0 with ~2% per-step reduction. Loss reaches approximately 1.0.
- Steps 36-40: Loss increases by ~0.1-0.15 per step (total increase > 0.5 over 5 steps), triggering the divergence condition. Gradient norms are elevated (5-8) but below the spike threshold.
- Steps 41+: Loss resumes decay, potentially at a reduced learning rate set by the recovery rule.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `stability_basics.loss_spike_recovery` | Loss increase > 0.5 over window of 5 steps | LR reduction / recovery action |

## Usage

```bash
hotcb scenario run scenarios/stability_divergence/
```
