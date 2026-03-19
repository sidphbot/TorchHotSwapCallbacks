# distill_warmup

Tests the `distillation_assist.summary_first_warmup` rule.

## Dynamics

- Steps 1-250: Slow, steady loss decay with `distill_weight=0.5` and `spatial_weight=0.3`.
- The `step` metric stays below 200 for most of the run, satisfying the warmup condition (`step < 200`).
- The rule triggers early and sets `distill_weight` to 0.9 to prioritize distillation during warmup.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `distillation_assist.summary_first_warmup` | `step < 200` | Set `distill_weight` to 0.9 |

## Usage

```bash
hotcb scenario run scenarios/distill_warmup/
```
