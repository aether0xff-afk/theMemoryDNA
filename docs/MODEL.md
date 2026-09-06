# Model provenance and extension boundary

This repository deliberately separates **Silva et al. (2021) biology/model equations** from the **new evolutionary-computation layer**.

## 1. Paper-derived layer

Primary source:

> Silva, W. T. A. F., Otto, S. P., & Immler, S. (2021). *Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions*. PLOS Genetics, 17(5), e1009581. https://doi.org/10.1371/journal.pgen.1009581

### sRNA dynamics — Eq. 1

\[
\frac{dn}{dt}=\left(\frac{b}{1+bn/m}-d\right)n+\mu
\]

Default values used in the paper's principal simulations:

- \(d=0.1\)
- \(m=5.0\)
- \(\mu=6.798\)
- \(c=20\) cell divisions per generation

### Somatic plasticity — Eq. 4 form

\[
b_t=b_0+P_b\,(b_{W\max,g}-b_0)(1-e^{-at})
\]

The paper studies both instantaneous plasticity and delayed plasticity with \(a=0.15\). This repository uses \(a=0.15\) by default.

### Direct transgenerational sRNA inheritance

\[
n_{initial,g+1}=r_{germ}\,n_{final,g}
\]

Silva et al. assume maternal transfer; this repository therefore uses the selected mother's adult sRNA state and her current \(r_{germ}\) when creating a child.

### Environment

The paper uses balanced environments \(\epsilon\in\{0.1,0.9\}\), each present 50% of the time, and controls their order through parent-offspring similarity \(p_\epsilon\). For \(G=20\), the paper's representative cases are approximately

\[
p_\epsilon\in\{0.11,0.53,0.89\}.
\]

Here

\[
p_\epsilon = 1-\frac{k}{G-1},
\]

where \(k\) is the number of switches between adjacent generations. The generator constructs exact balanced 10/10 cycles with the nearest feasible number of switches.

### Fitness — Eq. 7 implemented in log space

The model uses the logistic sRNA phenotype and the paper's default parameters
\(\alpha=15\), \(\beta=0.1\), \(h=5\), \(C_n=10^{-5}\), and \(C_b=50C_n\).

The implementation is validated against two numerical statements in the paper: the pointwise optimum is approximately \(n=2.85\) in \(\epsilon=0.1\) and \(n=7.19\) in \(\epsilon=0.9\), and strategy A reaches approximately \(n=58.8\) at adulthood for \(c=20\).

Lifetime fitness is a geometric mean over developmental time, implemented as a mean of log fitness.

## 2. New evolutionary-computation layer

Silva et al. compare predefined mutant strategies mainly through invasion fitness. This project instead maintains an explicit population of individuals

\[
I_i=(G_i,E_i)
\]

with

\[
G_i=(\mu_i,b_i,r_{germ,i},P_{b,i}),\qquad E_i=n_{initial,i}.
\]

The population undergoes:

1. development under Eq. 1,
2. fitness evaluation using the Silva fitness function,
3. tournament selection,
4. arithmetic crossover,
5. Gaussian mutation,
6. maternal sRNA transfer to offspring.

Thus, **selection/crossover/mutation are project additions**. The biological dynamics and transmission rule are not newly invented GA operators.

## 3. Ablations

| Model | genetic evolution | within-generation plasticity | cross-generation sRNA |
|---|---:|---:|---:|
| `ga` | yes | no | no |
| `plastic-ga` | yes | yes | no |
| `epi-ga` | yes | yes | yes |
| `fixed-epi` | genome fixed | yes | yes |

`fixed-epi` uses \(r_{germ}=0.11\) and \(P_b=0.8\) by default; both values are paper-motivated reference settings, not fitted to this project's results.

## 4. Main hypotheses tested

The code does **not** hard-code the hypothesis that stable environments must evolve a larger \(r_{germ}\). In fact, Silva et al. show that direct transcript inheritance and germline plasticity do not have identical dependence on environmental autocorrelation. The experiment asks whether a freely evolving coupled system creates a relationship among \(p_\epsilon\), \(r_{germ}\), and \(P_b\).

The regime-switch experiment additionally tests whether the epigenetic channel changes the time required for population fitness to recover after a change in environmental autocorrelation.
