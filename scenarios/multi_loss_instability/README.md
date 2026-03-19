# multi_loss_instability

## Policy Pack

`multi_loss_assist`

## Expected Rule

`multi_loss_assist.aux_instability_rollback`

## Dynamics

- `train_loss` decays normally from 2.0.
- `aux_loss` tracks normally for ~38 steps, then spikes sharply above 10.0
  around step 40, triggering `aux_instability_rollback`.
- After the spike, `aux_loss` slowly recovers but stays elevated.

## Expected Effect

The autopilot should roll back `aux_weight` to stabilize training when
`aux_loss` exceeds the instability threshold.

## Running

```bash
python train.py  # or invoke via scenario runner
```
