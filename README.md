# theMemoryDNA

A small evolutionary simulation connecting **small-RNA transgenerational inheritance** with a population-based evolutionary algorithm.

## Research question

> When parent and offspring environments differ in temporal similarity, does a population evolve different strengths of transgenerational small-RNA plasticity?

This repository deliberately separates two layers:

1. **Biological layer — literature-derived.** sRNA production/decay/amplification, developmental fitness, default constants, environmental stress values, `G=20`, and the slow-dynamics `0.75×` condition follow Silva, Otto & Immler (2021).
2. **Evolutionary-computation layer — our extension.** A finite population carries a mutable genetic parameter `P_b` and a maternally inherited epigenetic amplification state. Fitness-proportional selection and mutation repeatedly evolve `P_b` instead of testing one preselected mutant against one resident.

The point is **not** to claim that the GA hyperparameters are biological constants. They are computational experimental settings, while the molecular dynamics and fitness landscape are anchored to the published model.

## Biological model used

Silva et al. Eq. 1:

```text
dn/dt = ((b / (1 + b*n/m)) - d)*n + mu
```

Default values used in the paper and here:

- `mu = 6.798`
- `d = 0.1`
- `m = 5.0`
- `c = 20` cell divisions / generation (paper default)
- `alpha = 15`, `beta = 0.1`, `h = 5`
- `Cn = 1e-5`, `Cb = 50*Cn`
- benign/stress environments: `epsilon in {0.1, 0.9}` with 50% each
- parent-offspring environmental similarity labels: `p_epsilon in {0.11, 0.53, 0.89}`
- delayed plasticity constant: `a = 0.15`
- main evolutionary experiment: paper's slow-dynamics condition, multiplying `b,d,m,mu` by `0.75`

The implementation also uses the paper's Eq. 7 instantaneous fitness and Eq. 8A life geometric mean fitness.

### Sanity check

The paper reports that Strategy A with `b=0`, `mu=6.798`, `d=0.1`, `c=20` reaches adult `n ~ 58.8`. The numerical implementation reproduces this value before running the evolutionary extension.

## New evolutionary experiment

Each lineage has:

- **Genetic state:** `P_b in [0,1]`, the strength of the germline plastic response. It mutates and is selected.
- **Epigenetic state:** the maternal germline amplification state `b#`, inherited by the next generation and affecting the offspring soma before the offspring has reacted to its own environment.

The experiment is Strategy-F-like: current environmental information is written into the germline and becomes useful primarily if the offspring sees a similar environment. This is the situation for which Silva et al. predict strong dependence on environmental autocorrelation.

Selection is fitness-proportional. Reproduction is maternal/lineage-based with mutation rather than crossover so that the epigenetic state remains linked to the mother-offspring chain assumed by the biological model.

## Run

```bash
pip install -r requirements.txt
python run_experiments.py
```

Quick CI-sized run:

```bash
python run_experiments.py --quick --out ci-results
```

Results are written to `results/` as CSV, JSON, Markdown, and SVG.

## Main result

With 12 independent seeds, population 3000, 1500 evolutionary generations, and the paper's slow-dynamics condition (`0.75x`), evolved mean `P_b` was:

- `p_epsilon ~ 0.11`: `0.208 +/- 0.024`
- `p_epsilon ~ 0.53`: `0.233 +/- 0.052`
- `p_epsilon ~ 0.89`: `0.748 +/- 0.040`

So strong transgenerational germline plasticity evolved only in the highly autocorrelated environment. See [`results/REPORT.md`](results/REPORT.md) for details.

## Citation

Silva WTAF, Otto SP, Immler S. *Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions.* PLOS Genetics. 2021;17(5):e1009581. https://doi.org/10.1371/journal.pgen.1009581

## Interpretation boundary

This is an **in-silico evolutionary extension**, not an estimate of real *C. elegans* `P_b` values. The biological equations are literature-based, while population size, mutation strength, run length, and selection implementation are algorithmic choices introduced for this experiment.
