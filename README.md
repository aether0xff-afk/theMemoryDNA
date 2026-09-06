# theMemoryDNA — evolution of small-RNA memory strategies

Silva, Otto & Immler (2021)의 small-RNA 수리모델에 명시적 개체군 진화를 결합해, **생물이 이전 세대의 무엇을 기억하도록 진화하는가**를 실험하는 저장소입니다.

연구의 중심은 “epigenetic GA가 일반 GA보다 좋은가?”가 아닙니다. GA는 실험 장치이고, 최종 질문은 다음입니다.

> **환경의 시간적 예측 가능성이 달라질 때, 이전 세대의 sRNA 상태 자체를 전달하는 전략과 환경에 맞춰 조절된 sRNA 증폭 기작을 전달하는 전략은 어떻게 경쟁하고 공동진화하는가?**

## 생물학적 기반

sRNA abundance는 Silva Eq. 1을 사용합니다.

\[
\frac{dn}{dt}=\left(\frac{b}{1+bn/m}-d\right)n+\mu
\]

직접적인 transcript inheritance는

\[
n_{0,g+1}=r_{germ}n_{c,g}
\]

이고, delayed somatic plasticity는 Eq. 4를 사용합니다.

\[
b^*_{t,g+1}=b^\#_{c,g}+P_b\left(b_{Wmax,g+1}-b^\#_{c,g}\right)(1-e^{-at})
\]

새 핵심 실험에서는 Silva Eqs. 5-6의 **germline amplification-state inheritance**도 두 번째 epigenetic channel로 구현합니다.

\[
b^\#_{0,g+1}=b^\#_{c,g}
\]

\[
b^\#_{t,g+1}=b^\#_{c,g}+P_b\left(b_{Wmax,g+1}-b^\#_{c,g}\right)(1-e^{-at})
\]

중요하게도 별도의 임의 전달률 `r_b`는 만들지 않습니다. `b#` 전달은 논문의 상태 전달 규칙을 그대로 따릅니다.

자세한 식·해석 경계는 [`docs/MODEL.md`](docs/MODEL.md), 새 연구 설계는 [`docs/DUAL_MEMORY_STUDY.md`](docs/DUAL_MEMORY_STUDY.md)에 정리되어 있습니다.

## 1. 검증된 기준선: Genome + transcript Epigenome

기존 실험은 개체를

\[
I=(G,E),\qquad G=(\mu,b,r_{germ},P_b),\quad E=n_0
\]

로 두고 `ga`, `plastic-ga`, `epi-ga`, `fixed-epi`를 비교합니다.

이 단계는 Silva 모델 재현과 **세대 간 transcript channel 자체의 효과**를 확인하는 기준선입니다. corrected lifetime fitness는 zygote state인 `t=0`부터 적분하며, CI는 adult `n≈58.8`, Eq.7 pointwise optima, direct-transmission advantage를 회귀 테스트합니다.

실행:

```bash
python run_experiments.py --preset quick --seed 7 --output results/latest
```

## 2. 최종 연구: state memory vs mechanism memory

모든 실험군에 같은 within-generation somatic plasticity(Eq.4)를 허용하고, **세대 간 전달 채널만** 다르게 합니다.

| architecture | sRNA transcript state `n` | germline amplification state `b#` |
|---|---:|---:|
| `no-memory` | X | X |
| `state-memory` | O | X |
| `mechanism-memory` | X | O |
| `dual-memory` | O | O |

- `state-memory`: “지난 세대의 분자 상태/답을 기억”
- `mechanism-memory`: “그 상태를 만드는 환경조절 기작을 기억”
- `dual-memory`: 두 정보 채널을 동시에 사용

Genome은 여전히

\[
G=(\mu,b,r_{germ},P_b)
\]

이고, mechanism-memory에서는 epigenetic state에 zygotic `b#`가 추가됩니다.

## 최종 4개 실험

### Anchor

\[
p_\epsilon\in\{0.11,0.53,0.89\}
\]

4개 architecture를 paired seed로 비교합니다. Full preset: population 160, 500 generations, 20 replicates.

### Fine sweep

balanced 20-generation environment에서 실현 가능한 11개 `p_epsilon`을 약 0.05~0.89 범위에서 훑습니다.

### Matched-autocorrelation temporal pattern test

평균 환경, 10:10 상태 비율, switch 수와

\[
p_\epsilon=1-5/19\approx0.737
\]

을 모두 같게 유지하고 **run-length structure만** 바꿉니다.

- `regular`
- `clustered`
- `stochastic`

따라서 여기서 차이가 나면 단순한 parent-offspring similarity 하나로 설명할 수 없는 **고차 시간구조 효과**입니다.

### Regime switch

\[
0.89\rightarrow0.11\rightarrow0.89
\]

환경 변화 뒤 fitness뿐 아니라 `r_germ`, `P_b`, `mu`, inherited `n`, inherited `b#`가 어떤 순서로 재편되는지 추적합니다.

## 실행

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest -q
```

새 연구 component 실행:

```bash
python run_memory_study.py --component anchor   --preset quick --seed 7 --output results/anchor
python run_memory_study.py --component fine     --preset quick --seed 7 --output results/fine
python run_memory_study.py --component patterns --preset quick --seed 7 --output results/patterns
python run_memory_study.py --component switch   --preset quick --seed 7 --output results/switch
```

GitHub Actions의 **Dual Epigenetic Memory Study** workflow는 먼저 모든 회귀 테스트와 4-component smoke를 수행하고, main에서는 anchor/fine/patterns/switch full 실험을 병렬 실행해 artifact를 묶습니다.

## 해석 경계

- `P_b`는 Silva 식의 plasticity strength이며 transcript 자체가 아닙니다.
- 직접 transcript 전달과 germline amplification-state 전달은 별개의 정보 채널입니다.
- selection/crossover/mutation은 이 프로젝트의 evolutionary-computation layer입니다.
- 진화한 파라미터 절대값은 GA bounds와 mutation model에 조건부입니다.
- bet-hedging을 위한 offspring epigenetic variance evolution은 이번 본 연구에 섞지 않고 후속 연구로 남깁니다.

## References

1. Silva WTAF, Otto SP, Immler S. Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions. *PLOS Genetics*. 2021;17(5):e1009581. https://doi.org/10.1371/journal.pgen.1009581
2. Yuen S, Ezard THG, Sobey AJ. Epigenetic opportunities for evolutionary computation. *Royal Society Open Science*. 2023;10:221256. https://doi.org/10.1098/rsos.221256
