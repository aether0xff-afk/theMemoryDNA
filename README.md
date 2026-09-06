# theMemoryDNA — evolution of small-RNA memory strategies

Silva, Otto & Immler (2021)의 small-RNA 수리모델에 명시적 개체군 진화를 결합해, **변화하는 환경에서 생물이 이전 세대의 무엇을 기억하도록 진화하는가**를 조사하는 저장소입니다.

연구의 주인공은 GA가 아닙니다. GA는 진화 실험을 수행하기 위한 계산 도구이고, 최종 질문은 다음입니다.

> **환경의 시간적 구조가 달라질 때, sRNA를 새로 생산하는 전략, 이전 세대의 sRNA 상태를 직접 물려받는 전략, sRNA 증폭 기작의 상태를 물려받는 전략은 어떻게 경쟁하고 공동진화하는가?**

## 생물학적 기반

sRNA abundance는 Silva Eq. 1을 사용합니다.

\[
\frac{dn}{dt}=\left(\frac{b}{1+bn/m}-d\right)n+\mu
\]

직접적인 transcript inheritance는

\[
n_{initial,g+1}=r_{germ}n_{final,g}
\]

입니다. 이는 **지난 세대의 분자 상태(state)**를 직접 전달하는 채널입니다.

Silva는 somatic amplification \(b^*\)와 germline amplification \(b^\#\)를 구분하며, strategies E/F에서는 adult germline의 \(b^\#\)가 다음 세대에 전달됩니다. germline state의 환경 반응은 Eq. 6을 따릅니다.

\[
b^\#_{t,g+1}=b^\#_{c,g}+P_b\left(b_{W\max,g+1}-b^\#_{c,g}\right)(1-e^{-at})
\]

이 채널은 sRNA 양 자체가 아니라 **sRNA를 증폭하는 기작의 상태(mechanism state)**를 전달합니다.

중요하게도 별도의 임의 전달률 `r_b`는 만들지 않습니다. Silva Eqs. 5–6에서 \(b^\#\)는 상태 자체가 다음 세대로 전달되고, \(P_b\)가 현재 환경에 대한 조절 강도를 결정합니다.

자세한 출처와 해석 경계는 [`docs/MODEL.md`](docs/MODEL.md), 최종 연구 설계는 [`docs/FINAL_STUDY.md`](docs/FINAL_STUDY.md)에 정리되어 있습니다.

## 검증된 기준선

기존 `run_experiments.py`는 transcript-only epigenome을 가진 Epi-GA 기준선을 유지합니다. lifetime fitness는 zygote state인 \(t=0\)부터 적분하며, CI는 다음 Silva 기준을 검사합니다.

- strategy A adult sRNA \(n\approx58.8\)
- Eq. 7 pointwise optimum: \(n\approx2.85\) at \(\epsilon=0.1\), \(n\approx7.19\) at \(\epsilon=0.9\)
- direct transcript transmission의 early-life fitness advantage

기준선 실행:

```bash
python run_experiments.py --preset quick --seed 7 --output results/latest
```

## 최종 연구: state memory vs mechanism memory

최종 연구는 고정 amplification \(b=0\)으로 두어, transcription과 두 epigenetic memory channel을 분리합니다.

| architecture | direct sRNA state | inherited germline \(b^\#\) | somatic plasticity |
|---|---:|---:|---:|
| `genetic` | X | X | X |
| `somatic` | X | X | O — Silva D |
| `state-memory` | O — Silva C-like | X | X |
| `mechanism-memory` | X | O — Silva F | X |
| `dual-memory` | O | O — C+F | X |
| `full-dual-memory` | O | O — C+E | O |

### 두 가지 genetic regime

1. **fixed-μ:** \(\mu=6.798\)을 고정해 Silva의 resident 조건에서 memory architecture 자체를 비교합니다.
2. **evolved-μ:** \(0\le\mu\le10\)에서 transcription rate도 진화시켜 `de novo production ↔ inherited memory` trade-off를 조사합니다.

\(\mu\)가 진화할 때는 각 개체가 향하는 최적 amplification rate도 달라질 수 있으므로, \(b_{W\max}(\mu,\epsilon)\)를 미리 계산한 grid에서 개체별로 보간합니다. 즉 \(\mu=6.798\)에서 계산한 하나의 목표를 모든 genotype에 강제로 사용하지 않습니다.

## 최종 실험 3종

### A. Environmental-similarity sweep

\[
p_\epsilon\in\{0.11,0.53,0.89\}
\]

6개 architecture를 동일한 paired seed로 비교하고, fixed-μ와 evolved-μ를 분리합니다.

### B. Matched temporal-order test

20세대 cycle에서 benign/stressful을 10:10으로 유지하고 **within-cycle switch 수도 9회로 동일하게 고정**한 뒤, run-length arrangement만 바꿉니다.

- `periodic-2`
- `early-persistent`
- `late-persistent`
- `stochastic-matched`

따라서 여기서 차이가 나면 단순한 평균 환경이나 parent–offspring similarity만으로 설명되지 않는 **고차 시간 구조의 영향**입니다.

### C. Regime shift

\[
0.89\rightarrow0.11\rightarrow0.89
\]

fitness뿐 아니라 \(r_{germ}\), inherited \(b^\#\), \(P_b\), \(\mu\)의 재편 과정을 추적합니다.

## 실행

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest -q
```

최종 연구:

```bash
python run_final_study.py --preset quick --seed 7 --output results/final-study
```

Presets:

- `smoke`: CI용 end-to-end 확인
- `quick`: 그래프와 경향 확인
- `full`: population 140, 500 generations, architecture sweep 12 replicates, matched-pattern 8 replicates

생성물에는 raw generation history, replicate summary, paired effect table, matched-pattern 결과, regime-shift trace, PNG plots, metadata, `FINAL_RESULTS.md`가 포함됩니다.

## 해석 경계

- `P_b`는 responsiveness parameter이며, 그 자체가 전달되는 분자는 아닙니다.
- `n_initial`과 \(b^\#\)는 서로 다른 세대 간 정보 채널입니다.
- selection / crossover / mutation / evolving \(\mu\)는 이 프로젝트가 추가한 evolutionary-computation layer입니다.
- 진화한 parameter가 numerical bound에 닿으면 절대값을 생물학적 최적값으로 해석하지 않습니다.
- architecture 간 fitness 차이는 **이 모델의 조건에서** 얻은 결과이며 실제 생물 전체에 일반화하지 않습니다.
- offspring epigenetic variance를 진화시키는 bet-hedging은 본 연구에 섞지 않고 후속 연구로 남깁니다.

## References

1. Silva WTAF, Otto SP, Immler S. Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions. *PLOS Genetics*. 2021;17(5):e1009581. https://doi.org/10.1371/journal.pgen.1009581
2. Yuen S, Ezard THG, Sobey AJ. Epigenetic opportunities for evolutionary computation. *Royal Society Open Science*. 2023;10:221256. https://doi.org/10.1098/rsos.221256
