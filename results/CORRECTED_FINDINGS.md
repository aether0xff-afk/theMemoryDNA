# Corrected findings

> **Authoritative after lifetime-fitness integration fix.**
>
> Result files produced before commit `b2db467b1eee52c39d2dba4b029b3fcb905a426c` are invalid for scientific interpretation because the previous implementation omitted the zygotic/early-development interval from lifetime fitness.

## Provenance

- biological-model fix: `b2db467b1eee52c39d2dba4b029b3fcb905a426c`
- Strategy-C regression test: `cd2dd60ae4e158b130a78ea7e9f164037199e074`
- corrected full experiment: GitHub Actions run `34017755241`, artifact `9984540995`, seed 7
- corrected fine-sweep / plateau probe: run `34017762657`, artifact `9984482249`, seed 7
- corrected mu-bound sensitivity probe: run `34017775886`, artifact `9984504420`, seed 7

The corrected CI requires the model to reproduce the paper-motivated Strategy-C result: with fixed `mu=6.798`, `b=0`, and `P_b=0`, direct sRNA transmission must outperform `r_germ=0`, with an optimum in the neighborhood of `r_germ≈0.1`.

## 1. Main full experiment — 20 replicates

Final-window mean fitness (mean ± SD):

| p_epsilon | GA | Plastic-GA | Epi-GA | Fixed-Epi |
|---:|---:|---:|---:|---:|
| 0.11 | 0.49037 ± 0.00046 | 0.49023 ± 0.00085 | **0.52510 ± 0.03194** | **0.59367** |
| 0.53 | 0.46843 ± 0.00630 | 0.46798 ± 0.00711 | **0.52438 ± 0.02413** | **0.57410** |
| 0.89 | 0.46865 ± 0.00573 | 0.46856 ± 0.00539 | **0.50228 ± 0.01985** | **0.57446** |

Paired Epi-GA vs GA differences:

- `p_epsilon=0.11`: +0.03473, about **+7.1%**, paired t-test `p=1.11e-4`
- `p_epsilon=0.53`: +0.05594, about **+11.9%**, `p=8.39e-9`
- `p_epsilon=0.89`: +0.03363, about **+7.2%**, `p=1.51e-7`

Plastic-GA by itself did not differ detectably from GA in this experiment (`p=0.132`, `0.276`, `0.666` for the three environmental similarities). Thus the corrected experiment attributes the main performance difference to the **cross-generation sRNA channel**, not to within-generation amplification plasticity alone.

However, Fixed-Epi outperformed the evolving Epi-GA in every environment. Therefore the result is **not** evidence that the current GA finds the globally optimal epigenetic strategy. It shows that an inherited-sRNA channel can improve this evolutionary system while also revealing optimization/mutation-load or co-adaptation limitations in the current evolutionary layer.

## 2. What evolved inside Epi-GA?

| p_epsilon | mean r_germ | mean P_b | mean mu |
|---:|---:|---:|---:|
| 0.11 | 0.1262 ± 0.0474 | 0.0581 ± 0.0439 | 3.170 ± 0.833 |
| 0.53 | 0.1900 ± 0.0263 | 0.1844 ± 0.0603 | 2.130 ± 0.018 |
| 0.89 | 0.1287 ± 0.0172 | 0.4234 ± 0.0718 | 2.102 ± 0.026 |

`P_b` rose strongly and monotonically with environmental similarity. All paired differences among the three levels were significant (`p<1.5e-7`).

`r_germ`, in contrast, was **not monotonic**: it increased from 0.11 to 0.53 and then decreased again at 0.89. The 0.11 and 0.89 endpoints were not detectably different (`p=0.834`). Therefore the strong claim `higher p_epsilon -> higher r_germ` is rejected for the default joint-evolution configuration.

## 3. Fine sweep: no strong evidence for a biological r_germ plateau

The corrected 11-level fine sweep produced Epi-GA mean `r_germ` values from roughly 0.086–0.161. A linear model and a saturating model were essentially tied:

- linear AIC: `-898.87`
- saturating AIC: `-899.45`
- delta AIC (linear minus saturating): **0.57**

This difference is too small to support a meaningful plateau claim.

More importantly, when `mu=6.798` and baseline `b=0` were held fixed and the `(r_germ, P_b)` fitness landscape was scanned directly, the fitness-optimal direct transmission rate was approximately **`r_germ=0.10` across almost the whole p_epsilon range** (0.075 only at the highest grid point in this finite scan). This reproduces the Strategy-C-like direct-transmission optimum and shows that direct transcript inheritance itself does not need to increase monotonically with environmental autocorrelation.

## 4. Production–inheritance trade-off and the mu-bound sensitivity

The absolute evolved value of `r_germ` is highly sensitive to how low the GA is allowed to evolve de-novo transcription `mu`.

At `p_epsilon≈0.895`:

| mu lower bound | evolved mean mu | evolved mean r_germ |
|---:|---:|---:|
| 0 | 0.316 | **0.772** |
| 1 | 1.104 | **0.250** |
| 2 (default) | 2.124 | **0.142** |
| 3 | 3.138 | **0.101** |

With `mu_min=0`, the increase in `r_germ` continued strongly to high environmental similarity and a linear model fit better than the saturating alternative. With higher lower bounds, the direct-transmission response became progressively flatter.

This supports a **co-evolutionary trade-off** interpretation: if individuals can reduce de-novo sRNA production, they can compensate by relying more strongly on inherited sRNA. Consequently, absolute `r_germ` values from a freely evolving `(mu,b,r_germ,P_b)` genome are not biological estimates and must always be reported together with the allowed genetic parameter ranges.

## 5. Regime-shift test

Under the current recovery metric, Epi-GA did **not** recover faster than the controls after `0.89 -> 0.11 -> 0.89` changes.

- `0.89 -> 0.11`: Epi-GA mean recovery = 11.1 generations, median = 0; the large variance makes the paired comparison with the zero-lag controls non-significant at 0.05 (`p≈0.063`).
- `0.11 -> 0.89`: Epi-GA mean recovery = 1.95 generations versus 0 for the other three groups (`p≈2.16e-9`).

Therefore the hypothesis that the epigenetic channel necessarily reduces post-shift recovery time is **not supported** by this implementation.

## 6. Interpretation boundary

The inherited epigenetic state in this repository is **directly transmitted sRNA abundance `n`** via

`n_initial,g+1 = r_germ * n_final,g`.

`P_b` is the within-generation somatic amplification plasticity based on Silva Eq. 4. The project does **not yet implement a second inherited amplification-state `b#` channel corresponding to the full Strategy-E/F germline machinery**. Claims about the present results should therefore be phrased as evolution of direct sRNA inheritance plus somatic amplification plasticity, not as a complete reproduction of every transgenerational mechanism in Silva et al.

## Current best research statement

> Using Silva et al.'s small-RNA dynamics and direct maternal sRNA-transfer rule as the biological layer, a population-based evolutionary algorithm evolved genetic production, amplification, plasticity and direct inheritance parameters. After correcting lifetime-fitness integration and validating the Strategy-C baseline, direct sRNA inheritance improved final population fitness over genetic-only and plasticity-only controls, while the evolved balance between de-novo production and inherited sRNA depended strongly on environmental similarity and on the allowed genetic parameter range. Within-generation plasticity `P_b` increased strongly with environmental similarity, whereas direct transmission `r_germ` did not follow a simple monotonic relationship unless de-novo production was allowed to fall substantially.
