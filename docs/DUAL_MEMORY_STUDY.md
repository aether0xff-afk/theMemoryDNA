# Legacy prototype — superseded

> **Do not use this document as the final study protocol.**
>
> This was an earlier dual-memory prototype. The canonical implementation and protocol are now:
>
> - [`docs/FINAL_STUDY.md`](FINAL_STUDY.md)
> - `src/memorydna/dual_memory.py`
> - `src/memorydna/final_study.py`
> - `run_final_study.py`
>
> The prototype is retained only for provenance. It shared somatic plasticity across all architectures and used a single resident-parameter `b_Wmax` target even while `mu` could evolve. The final study instead separates Silva D/E/F architectures explicitly and, in the evolved-`mu` condition, uses genotype-specific `b_Wmax(mu, epsilon)` lookup/interpolation.

## Historical research question

The GA was treated as an experimental instrument rather than the endpoint. This prototype asked whether transmitting the previous sRNA molecular state differs from transmitting the germline amplification state `b#`.

## Why it was superseded

The prototype was useful for establishing the state-vs-mechanism distinction, but two design choices made causal interpretation weaker:

1. All memory architectures retained within-generation somatic Eq. 4 plasticity, making the contribution of the inherited mechanism state harder to isolate.
2. The resident `b_Wmax` calculated at Silva's default `mu=6.798` was used even when `mu` evolved, although changing transcription can change the amplification optimum.

The final study corrects both issues and adds explicit paper-bound (`mu=6.798`) and co-evolution (`mu` evolves) regimes.

## Primary source

Silva WTAF, Otto SP, Immler S. Evolution of plasticity in production and transgenerational inheritance of small RNAs under dynamic environmental conditions. *PLOS Genetics*. 2021;17(5):e1009581. https://doi.org/10.1371/journal.pgen.1009581
