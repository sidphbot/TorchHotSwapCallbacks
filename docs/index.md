# hotcb

<div style="text-align: center; padding: 2rem 0;">
<h2 style="font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.5rem;">Live Training Control Plane for PyTorch</h2>
<p style="font-size: 1.15rem; color: #888; max-width: 600px; margin: 0 auto;">
Swap callbacks, tune hyperparameters, adjust loss weights, and run AI autopilot — all while your model trains. No restart, no lost progress.
</p>
</div>

---

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.5rem; margin: 2rem 0;">

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
<h3 style="margin-top: 0;">Live Mutation</h3>
<p style="color: #aaa; font-size: 0.9rem;">Swap callbacks, tune optimizer params, adjust loss weights mid-run. Changes apply at the next training step — zero downtime.</p>
<p><strong>98.66% → 98.88%</strong> MNIST via late-stage LR reduction</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
<h3 style="margin-top: 0;">Autopilot</h3>
<p style="color: #aaa; font-size: 0.9rem;">5 policy packs detect instability, plateaus, loss conflicts. Rule-based or LLM-driven — choose your level of automation.</p>
<p><strong>12 scenarios</strong> validated, one per policy rule</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
<h3 style="margin-top: 0;">Continuation Tuning</h3>
<p style="color: #aaa; font-size: 0.9rem;">Branch from converged checkpoints, apply small mutations, keep only what improves. Cheap local search around good basins.</p>
<p><strong>+6.4% CIFAR-10</strong> from 500 extra steps</p>
</div>

<div style="border: 1px solid #333; border-radius: 12px; padding: 1.5rem; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
<h3 style="margin-top: 0;">Dashboard</h3>
<p style="color: #aaa; font-size: 0.9rem;">Real-time metric charts, manifold visualizations, research graph, recipe editor. Four themes, keyboard navigation, WebSocket streaming.</p>
<p><code>hotcb serve --dir runs/exp1</code></p>
</div>

</div>

---

## Proof: Late-Stage Continuation Results

Continuation tuning from converged baselines — every branch improved over the base run:

| Recipe | MNIST (14K CNN) | CIFAR-10 (62K CNN) |
|--------|:-:|:-:|
| **lr_half** | 98.66% → 98.85% (+0.19%) | 68.86% → **73.26% (+6.39%)** |
| **lr_quarter** | 98.66% → **98.88% (+0.22%)** | 68.86% → 73.07% (+6.11%) |
| **swa_tail** | 98.66% → 98.85% (+0.19%) | 68.86% → 72.77% (+5.68%) |
| **ema_tail** | 98.66% → 98.70% (+0.04%) | 68.86% → 72.00% (+4.56%) |
| **grad_clip_tight** | 98.66% → 98.73% (+0.07%) | 68.86% → 73.18% (+6.27%) |
| **wd_reduce** | 98.66% → 98.74% (+0.08%) | 68.86% → 71.45% (+3.76%) |

All experiments reproducible with two commands:

```bash
hotcb continue baseline --task cifar10 --out runs/cifar10_baseline
hotcb continue run --task cifar10 --run runs/cifar10_baseline --metric val_accuracy --mode max
```

[Full results and methodology →](results/continuation.md)

---

## Quick Start

```bash
pip install -e ".[dev,all]"
hotcb demo                      # synthetic training + live dashboard
hotcb demo --golden             # multi-task demo with rich metrics
```

## Documentation

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0;">

<div>

**Get Started**

- [Getting Started](getting-started.md)
- [Concepts](concepts.md)
- [CLI Reference](cli.md)

</div>
<div>

**Features**

- [Autopilot](autopilot.md)
- [Policy Packs](policy_packs.md)
- [Continuation Tuning](features/continuation.md)

</div>
<div>

**Results**

- [Continuation Results](results/continuation.md)
- [Policy Improvements](results/policies.md)
- [Scenarios](scenarios.md)

</div>
<div>

**Reference**

- [Modules](modules/cb.md)
- [File Formats](formats.md)
- [API Reference](api/index.md)

</div>

</div>
