# finish_lr_reduction

Tests the `finish_strong.mutation_lockdown` rule.

## Dynamics

- Steps 851-900: Late-stage training with slow loss decay. Step counter starts at 850 to simulate a run nearing completion (total ~1000 steps).
- Step 900: The `mutation_lockdown` rule fires (step > 900), applying `lr_mult: 0.1` to stabilize the final convergence phase.
- Steps 900-1000: Training continues with reduced LR for final fine-tuning.

## Expected rule trigger

| Rule | Trigger condition | Action |
|------|-------------------|--------|
| `finish_strong.mutation_lockdown` | `step > 900` | `lr_mult: 0.1` |

## Usage

```bash
hotcb scenario run scenarios/finish_lr_reduction/
```
