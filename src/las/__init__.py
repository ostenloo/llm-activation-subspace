"""Implementation of the pre-registration in SPEC.md.

Module map, by spec section:
  config    §2, §3, §8   frozen constants
  subspace  §2, §3, §6   PCA subspaces, the capture statistic A, the estimand I
  controls  §7           absolute floors and the rogue-dimension diagnostic
  gate      §8           split-half identifiability gate
  corpus    §5           XSTest pair selection and the asymmetric focus-clean split

Nothing here extracts activations or computes A on real data; SPEC.md §10 gates
that on OPEN-1/2/4/5/6 being closed in writing.
"""
