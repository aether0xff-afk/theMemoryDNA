# Final study: state memory vs mechanism memory

## Research question

> When environmental predictability changes across generations, how do organisms allocate information between de novo sRNA production, direct inheritance of the previous generation's sRNA state, and inheritance of the machinery state that controls sRNA amplification?

The genetic algorithm is the experimental method, not the biological claim. All biological state updates below are taken from Silva, Otto & Immler (2021); selection, crossover, mutation, and allowing selected parameters to evolve are the new evolutionary-computation layer.

## Paper-derived memory channels

### 1. State memory: direct sRNA inheritance

Silva strategy C transmits a fraction of maternal adult sRNA to the zygote:

\[
n_{initial,g+1}=r_{germ}n_{final,g}.
\]

This transfers a **phenotypic state** that can improve fitness immediately after fertilization.

### 2. Mechanism memory: inherited germline amplification state

Silva distinguishes somatic amplification \(b^*\) from germline amplification \(b^\#\). For strategies E and F, the adult germline amplification state is inherited by the next generation. During development the germline value changes according to Eq. 6:

\[
b^\#_{t,g+1}=b^\#_{c,g}+P_b\left(b_{W\max,g+1}-b^\#_{c,g}\right)(1-e^{-at}).
\]

The offspring therefore receives not an sRNA quantity but an **amplification state shaped by the parental lineage**.

There is deliberately **no invented `r_b` parameter** in this implementation. Silva Eqs. 5-6 transmit \(b^\#\) directly; \(P_b\) controls the degree to which the germline amplification state responds to the current environment.

### 3. Somatic plasticity control

Silva strategy D resets the amplification state each generation and changes only the soma:

\[
b^*_t=0+P_b(b_{W\max}-0)(1-e^{-at}).
\]

This provides a within-generation response without mechanism inheritance.

## Architectures

| Architecture | direct sRNA state | inherited germline \(b^\#\) | current somatic plasticity |
|---|---:|---:|---:|
| `genetic` | no | no | no |
| `somatic` | no | no | yes (strategy D) |
| `state-memory` | yes | no | no (strategy C-like) |
| `mechanism-memory` | no | yes | no (strategy F) |
| `dual-memory` | yes | yes | no (C + F) |
| `full-dual-memory` | yes | yes | yes (C + E) |

Fixed amplification \(b\) is held at zero in the final architecture comparison. This avoids adding a fourth competing route and isolates transcription, state memory, and amplification-state memory.

## Two genetic regimes

### Paper-bound regime

\[
\mu=6.798
\]

is fixed. This asks which memory architecture is favored without changing the resident transcription rate used by Silva.

### Co-evolution regime

\[
0\le\mu\le10
\]

is allowed to evolve. Because the optimal amplification rate depends on \(\mu\), the implementation recomputes \(b_{W\max}(\mu,\epsilon)\) on a grid and interpolates it for each individual. This prevents the evolved-\(\mu\) experiment from using an optimum calculated only for \(\mu=6.798\).

## Experiments

### A. Environmental similarity sweep

\[
p_\epsilon\approx0.11,0.53,0.89.
\]

All six architectures are run with paired random seeds in both fixed- and evolved-\(\mu\) regimes.

Primary measurements:

- mean fitness
- \(\mu\)
- \(r_{germ}\)
- \(P_b\)
- inherited sRNA \(n_{initial}\)
- inherited amplification state \(b^\#\)

### B. Same first-order similarity, different temporal order

Silva reports that, once germline inheritance is present, the exact order of environments can matter even when the environmental composition and \(p_\epsilon\) are held fixed.

The study therefore compares balanced 20-generation cycles that all contain ten benign and ten stressful generations and nine within-cycle switches, but have different run-length arrangements:

- `periodic-2`
- `early-persistent`
- `late-persistent`
- `stochastic-matched`

This separates first-order parent-offspring similarity from higher-order temporal structure.

### C. Regime shift

\[
0.89\rightarrow0.11\rightarrow0.89
\]

The analysis follows not only fitness but also the trajectories of \(r_{germ}\), inherited \(b^\#\), \(P_b\), and \(\mu\).

## Interpretation boundaries

1. `P_b` is a responsiveness parameter and incurs the Silva plasticity cost. It is not itself an inherited molecule.
2. `b^#` is the inherited mechanism state.
3. `r_germ` transfers sRNA molecules directly and is biologically distinct from `b^#` inheritance.
4. GA operators are project additions and should not be described as biological equations from Silva.
5. If an evolved parameter hits a numerical bound, the result must be treated as bound-sensitive until a sensitivity analysis confirms otherwise.
6. A higher fitness for an architecture is evidence within this model, not proof that the corresponding mechanism is universally adaptive in living organisms.

## Reference

Silva WTAF, Otto SP, Immler S. Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions. *PLOS Genetics*. 2021;17(5):e1009581. doi:10.1371/journal.pgen.1009581.
