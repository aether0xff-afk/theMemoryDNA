# Dual epigenetic-memory study

## Research question

The GA is an experimental instrument, not the research endpoint. The biological question is:

> When environmental information persists across generations, is it more useful to transmit the previous molecular state itself, or to transmit an environmentally adjusted mechanism that can regenerate that state?

This study separates two small-RNA inheritance channels from Silva, Otto & Immler (2021):

1. **State memory** — direct transmission of sRNA transcripts.
2. **Mechanism memory** — transmission of the germline amplification state `b#`.

## Paper-derived equations

sRNA abundance follows Silva Eq. 1:

\[
\frac{dn}{dt}=\left(\frac{b}{1+bn/m}-d\right)n+\mu.
\]

Direct transcript inheritance is

\[
n_{0,g+1}=r_{germ}n_{c,g}.
\]

All architectures share delayed somatic plasticity (Eq. 4):

\[
b^*_{t,g+1}=b^\#_{c,g}+P_b\left(b_{Wmax,g+1}-b^\#_{c,g}\right)(1-e^{-at}).
\]

For the mechanism-memory architectures, the zygote inherits the adult germline amplification state and that germline state is updated according to Silva Eqs. 5-6:

\[
b^\#_{0,g+1}=b^\#_{c,g},
\]

\[
b^\#_{t,g+1}=b^\#_{c,g}+P_b\left(b_{Wmax,g+1}-b^\#_{c,g}\right)(1-e^{-at}).
\]

There is deliberately **no invented `r_b` parameter**. Silva models amplification-state inheritance as transmission of `b#` itself.

## Four matched architectures

All four permit the same within-generation Eq. 4 somatic plasticity. Only cross-generation channels differ.

| architecture | transcript state `n` | amplification state `b#` |
|---|---:|---:|
| `no-memory` | no | no |
| `state-memory` | yes | no |
| `mechanism-memory` | no | yes |
| `dual-memory` | yes | yes |

Operationally, `no-memory` is strategy-D-like, `mechanism-memory` is strategy-E-like, and state/dual architectures correspond to adding direct transcript transmission to those backgrounds, a combination also examined in Silva supplementary analysis (S9).

## Evolutionary layer

Each population evolves the genetic variables

\[
G=(\mu,b,r_{germ},P_b),
\]

subject to the architecture constraint that `r_germ=0` when direct transcript inheritance is unavailable. Selection, arithmetic crossover and Gaussian mutation are project additions; the molecular dynamics and cross-generational state updates are paper-derived.

## Experiments

### A. Anchor comparison

\[
p_\epsilon\in\{0.11,0.53,0.89\}
\]

All four architectures are compared with paired random seeds. Full preset: population 160, 500 generations, 20 replicates.

### B. Fine autocorrelation sweep

Eleven exactly realizable balanced 20-generation cycles are used from approximately 0.05 to 0.89. Full preset: 360 generations, 10 replicates.

### C. Matched-autocorrelation pattern test

This is the main extension beyond a simple `p_epsilon` sweep. Every cycle contains 10 benign and 10 stressful generations and exactly five switches, so

\[
p_\epsilon=1-5/19\approx0.737.
\]

Only run-length structure differs:

- `regular`: runs are distributed as evenly as possible;
- `clustered`: long blocks plus short one-generation runs;
- `stochastic`: random positive run-length composition.

Thus any difference cannot be attributed to mean environment, state frequency, or first-order parent-offspring similarity.

### D. Regime switch

\[
0.89\rightarrow0.11\rightarrow0.89.
\]

Rather than reducing the result to one recovery-time number, the analysis tracks the reorganization of fitness, `r_germ`, `P_b`, `mu`, inherited `n`, and inherited zygotic `b`.

## Interpretation boundaries

- `P_b` is a plasticity-strength parameter in the Silva equations; it is not itself an inherited transcript.
- `b#` inheritance and transcript inheritance are distinct molecular information channels.
- The absolute evolved values of `mu`, `b`, `r_germ`, and `P_b` remain conditional on the GA parameter bounds and mutation model.
- A higher-fitness architecture does not prove that the corresponding biological mechanism is universally superior; it is a result under the specified Silva fitness function and environmental process.
- Bet-hedging via evolution of offspring epigenetic variance is intentionally left for follow-up work so that this study retains one primary question.

## Primary source

Silva WTAF, Otto SP, Immler S. Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions. *PLOS Genetics*. 2021;17(5):e1009581. https://doi.org/10.1371/journal.pgen.1009581
