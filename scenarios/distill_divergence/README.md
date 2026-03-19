# distill_divergence

Tests the `distillation_assist.temperature_guard` rule.

## Dynamics

- Steps 1-54: Normal distillation training with gradual loss decay. `distill_loss` starts at 5.0 and decays slowly.
- Steps 55-74: `distill_loss` spikes to 22-27, well above the threshold of 20.0. `train_loss` destabilizes.
- Steps 75+: `distill_loss` recovers toward normal range after the rule fires.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `distillation_assist.temperature_guard` | `distill_loss > 20.0` | Set `distill_weight` to 0.3 |

## Usage

```bash
hotcb scenario run scenarios/distill_divergence/
```
