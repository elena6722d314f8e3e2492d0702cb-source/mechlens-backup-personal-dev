"""Checkpoint-based Command-R architecture acceptance against Hugging Face.

Uses a small pretrained Cohere checkpoint so the test fits on CPU. This verifies
the Command-R architecture, not the quality of the full-size Command-R models.
Run explicitly: uv run pytest tests/acceptance/model_bridge/test_command_r.py
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def command_r_models():
    """Load independent checkpoint copies with identical precision and attention."""
    reference = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    bridge.eval()
    return bridge, reference


def test_command_r_logits_match_hugging_face(command_r_models):
    """Every raw logit must differ from HF by less than 1e-4."""
    bridge, reference = command_r_models
    # Identical token IDs isolate model numerics from tokenizer/BOS policy.
    # Multiple nonzero positions exercise the interleaved rotary convention.
    tokens = torch.tensor(
        [[1, 7, 3, 9, 6, 12, 4, 15], [1, 2, 4, 8, 5, 11, 3, 14]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        expected = reference(input_ids=tokens, use_cache=False).logits
        actual = bridge(tokens, use_cache=False)

    assert actual.shape == expected.shape
    assert torch.isfinite(expected).all(), "HF reference produced non-finite logits"
    assert torch.isfinite(actual).all(), "Bridge produced non-finite logits"
    max_abs_diff = (actual - expected).abs().max().item()
    assert max_abs_diff < 1e-4, (
        f"{MODEL}: maximum absolute logit difference {max_abs_diff:.8g} "
        "exceeds the strict 1e-4 parity threshold"
    )
