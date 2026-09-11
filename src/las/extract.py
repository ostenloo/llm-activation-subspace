"""Activation extraction (SPEC.md §5, §10 step 2).

The only module here that imports torch. Everything else in `las` is numpy, so
the analysis runs without a GPU.

Position convention: SPEC.md §5 fixes the probe at `-5`, the `<|eot_id|>` closing
the user turn -- the first position to have attended over the complete user
message. Positions -4..-1 are fixed assistant-header template tokens, identical
for every prompt. Batches are LEFT-padded so that a negative index counts back
from the true end of each sequence rather than from padding.
"""

from __future__ import annotations

import dataclasses

import numpy as np

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PROBE_POSITION = -5


@dataclasses.dataclass
class Loaded:
    model: object
    tok: object


def load(device: str = "cuda") -> Loaded:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok.padding_side = "left"          # see module docstring
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return Loaded(model=model, tok=tok)


def chat_format(tok, prompts: list[str]) -> list[str]:
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts]


def assert_probe_token(tok, texts: list[str], position: int = PROBE_POSITION) -> None:
    """Fail loudly if the probe is not landing on <|eot_id|> (§5)."""
    want = tok.convert_tokens_to_ids("<|eot_id|>")
    for t in texts:
        got = tok(t, add_special_tokens=False)["input_ids"][position]
        if got != want:
            raise AssertionError(
                f"probe position {position} is token {got} ({tok.decode([got])!r}), "
                f"expected <|eot_id|>. The template changed; §5 must be revisited."
            )


def capture(L: Loaded, prompts: list[str], batch_size: int = 16,
            position: int = PROBE_POSITION) -> np.ndarray:
    """Activations at every layer for one probe position.

    Returns float32 [n_layers, n_prompts, hidden]. The float32 cast happens here:
    bf16 ties would break the PCA downstream.
    """
    import torch

    texts = chat_format(L.tok, prompts)
    assert_probe_token(L.tok, texts, position)

    out = []
    for i in range(0, len(texts), batch_size):
        enc = L.tok(texts[i:i + batch_size], return_tensors="pt",
                    padding=True, add_special_tokens=False).to(L.model.device)
        with torch.no_grad():
            hs = L.model(**enc, output_hidden_states=True).hidden_states
        # hidden_states[0] is the embedding output; layers 1..32 follow.
        layers = torch.stack(hs[1:], dim=0)          # [32, batch, seq, hidden]
        out.append(layers[:, :, position, :].float().cpu().numpy())
    return np.concatenate(out, axis=1)


def release(L: Loaded) -> None:
    """Free the model's VRAM.

    Extraction is followed by a long numpy-only phase (the gate, the bootstrap).
    Holding the model through it strands ~25 GB and OOMs anything else on the box.
    """
    import gc

    import torch

    L.model = None
    gc.collect()
    torch.cuda.empty_cache()
