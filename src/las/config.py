"""Frozen constants from SPEC.md. Changing one of these is a deviation (§11)."""

# §2
D_MODEL = 4096
N_LAYERS = 32

# §3
VARIANCE_LEVELS = (0.50, 0.75, 0.90)
LOGIT_EPS = 1e-6

# §7 -- rogue-set size for the ablated gate, with the reported sensitivities.
ROGUE_K = 5
ROGUE_K_SENSITIVITY = (1, 3)

# §8
GATE_THRESHOLD = 0.70
GATE_SIM_N = 225

# §5 -- recommended in OPEN-1, not yet closed.
D_MAX_RECOMMENDED = 3

PREPROCESSING = ("center", "standardize")  # primary, sensitivity (§6)
