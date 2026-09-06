# theMemoryDNA — small-RNA Epigenetic GA

Silva et al. (2021)의 small RNA 수리모델을 **개체군 수준의 유전 알고리즘** 위에 올려, 유전 정보와 후성유전 정보가 동시에 전달될 때 어떤 전략이 진화하는지 실험하는 저장소입니다.

핵심 질문은 단순 재현이 아니라 다음입니다.

> **환경의 세대 간 유사도 \(p_\epsilon\)가 달라질 때, small-RNA 전달률 \(r_{germ}\)과 가소성 \(P_b\)은 어떤 값으로 진화하는가?**

그리고 환경을 중간에 `0.89 → 0.11 → 0.89`로 바꾸어,

> **후성유전 채널이 환경 변화 뒤 적응도의 회복 시간을 줄이는가?**

도 함께 측정합니다.

## 구조

각 디지털 개체는

\[
\text{Individual}=(\text{Genome},\text{Epigenome})
\]

으로 표현합니다.

- Genome: \((\mu,b,r_{germ},P_b)\)
- Epigenome: 자손이 부모로부터 물려받은 초기 sRNA 양 \(n_{initial}\)

sRNA lifetime dynamics는 Silva et al. Eq. 1을 그대로 사용합니다.

\[
\frac{dn}{dt}=\left(\frac{b}{1+bn/m}-d\right)n+\mu
\]

직접적인 세대 간 전달 역시 논문의 식을 사용합니다.

\[
n_{initial,g+1}=r_{germ}\,n_{final,g}
\]

반면 **population, tournament selection, crossover, mutation**은 이 프로젝트가 추가한 evolutionary-computation layer입니다. 자세한 출처/경계는 [`docs/MODEL.md`](docs/MODEL.md)에 정리했습니다.

## 4개 실험군

| 모델 | genetic evolution | lifetime plasticity | intergenerational sRNA |
|---|---:|---:|---:|
| `ga` | O | X | X |
| `plastic-ga` | O | O | X |
| `epi-ga` | O | O | O |
| `fixed-epi` | genome 고정 | O | O |

따라서

- `ga` vs `plastic-ga` → 현세대 환경 반응의 효과
- `plastic-ga` vs `epi-ga` → sRNA를 자손에게 전달하는 효과
- `epi-ga` vs `fixed-epi` → 후성유전 기작 자체가 진화하는 효과

를 분리해서 볼 수 있습니다.

## 논문 기반 기본값

- \(d=0.1\)
- \(m=5.0\)
- \(\mu=6.798\)
- 세대당 20 cell divisions
- \(\epsilon\in\{0.1,0.9\}\), 50:50
- \(p_\epsilon\approx0.11,0.53,0.89\)
- delayed plasticity: \(a=0.15\)
- fitness: \(\alpha=15,\beta=0.1,h=5,C_n=10^{-5},C_b=50C_n\)

테스트에서는 논문이 보고한 기준값인 `adult n ≈ 58.8`, 순간 fitness optimum `n ≈ 2.85 (ε=0.1)`, `n ≈ 7.19 (ε=0.9)`도 검사합니다.

## 실행

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest -q
```

빠른 실험:

```bash
python run_experiments.py --preset quick --seed 7 --output results/latest
```

설정:

- `smoke`: CI용 아주 작은 실행
- `quick`: 그래프 형태를 확인하기 위한 반복 5회
- `full`: population 180, static 600 generations, 반복 20회

## 생성 결과

`results/<name>/`에 다음이 저장됩니다.

- `static_history.csv` — 모든 세대의 fitness, \(\mu,b,r_{germ},P_b,n\)
- `static_summary.csv` — 각 \(p_\epsilon\) × 모델 × 반복의 마지막 구간 요약
- `static_fitness.png` — ablation fitness 비교
- `static_evolved_r.png` — \(p_\epsilon\)별 진화한 \(r_{germ}\)
- `static_evolved_plasticity.png` — \(p_\epsilon\)별 진화한 \(P_b\)
- `switch_history.csv` — `0.89 → 0.11 → 0.89` 환경 변화 실험
- `switch_recovery.csv` — 각 변화 뒤 fitness 회복 세대 수
- `switch_fitness.png`
- `switch_r_trace.png`
- `metadata.json` — 실험 파라미터와 실제 생성된 환경 조건

## 주의할 해석

이 프로젝트는 `r_germ`이 안정 환경에서 반드시 커진다고 코딩하지 않습니다. Silva 논문에서도 **직접적인 sRNA transcript 전달**과 **환경에 반응하는 germline plasticity**의 환경 자기상관 의존성은 동일하지 않습니다. 따라서 `r_germ`과 `P_b`가 같이 진화했을 때 기대한 trade-off가 나타나는지, 또는 오히려 `r_germ`은 거의 일정하고 `P_b`가 변하는지를 결과로 판단해야 합니다.

## References

1. Silva WTAF, Otto SP, Immler S. Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions. *PLOS Genetics*. 2021;17(5):e1009581. https://doi.org/10.1371/journal.pgen.1009581
2. Yuen S, Ezard THG, Sobey AJ. Epigenetic opportunities for evolutionary computation. *Royal Society Open Science*. 2023;10:221256. https://doi.org/10.1098/rsos.221256
