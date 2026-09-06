# Preliminary quick-run notes

These are **exploratory results**, not the final confirmatory experiment.

Run configuration:

```bash
python run_experiments.py --preset quick --seed 7 --output <dir>
```

The `quick` preset uses population 80, 5 replicate seeds, 240 generations per static environment, and 120 generations per regime-shift phase.

## Static environmental-similarity sweep

For the freely evolving `epi-ga`, the final-window mean `r_germ` increased with environmental similarity:

| target p_epsilon | evolved mean r_germ | SD | evolved mean P_b | mean fitness |
|---:|---:|---:|---:|---:|
| 0.11 | 0.0197 | 0.0046 | 0.1718 | 0.5940 |
| 0.53 | 0.0344 | 0.0169 | 0.1202 | 0.5635 |
| 0.89 | 0.0949 | 0.0144 | 0.2870 | 0.5021 |

Thus the quick experiment shows the predicted **directional association** between parent-offspring environmental similarity and evolved direct sRNA transmission rate.

However, this is not evidence that the epigenetic channel improved performance. At `p_epsilon=0.89`, `epi-ga` mean fitness was ~0.502, whereas the genetic-only control was ~0.572. The evolved mean transcription rate `mu` also fell to ~2.14 in `epi-ga`, suggesting a strong coupled evolutionary response between inherited sRNA and de novo production.

This is an important negative/qualifying result: the current implementation does **not** force epigenetic inheritance to be beneficial, and the quick run does not support the simple hypothesis that adding the epigenetic channel necessarily raises fitness.

## Regime shift

Using the current 90%-of-phase-tail recovery metric, the `epi-ga` population required on average:

- `p_epsilon 0.89 -> 0.11`: 14.8 generations (SD 19.5)
- `p_epsilon 0.11 -> 0.89`: 1.0 generation (SD 1.2)

The other three ablation groups returned zero generations under this metric because their mean fitness did not drop below the phase-specific recovery threshold at the switches. Therefore the original hypothesis that epigenetic inheritance shortens recovery time is **not supported by this quick run**.

## What should be treated as the main result?

Not yet the absolute fitness ranking. The robust question for the full run is whether the increase in evolved `r_germ` with `p_epsilon` survives more generations, a larger population, and more replicate seeds. The full preset is provided specifically for that confirmatory step.
