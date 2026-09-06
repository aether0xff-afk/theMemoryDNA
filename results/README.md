# Result validity

Do not use result files produced before commit `b2db467b1eee52c39d2dba4b029b3fcb905a426c` for scientific interpretation.

That commit fixes lifetime-fitness integration so that Silva Eq. 8A includes the zygotic/early-development state at `t=0`. Omitting that interval artificially removed the principal early-life benefit of directly inherited small RNA and caused the implementation to contradict Silva et al.'s Strategy C result.

The corrected model is guarded by a regression test requiring direct sRNA transmission at fixed `mu=6.798`, `b=0`, and `P_b=0` to outperform `r_germ=0`, with an optimum in the neighborhood of `r_germ≈0.1`.

Authoritative experiment outputs should come from GitHub Actions runs using the corrected model and should be identified by their workflow run ID, commit SHA, seed, and metadata file.
