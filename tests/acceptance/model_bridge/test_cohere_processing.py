"""Offline Command-R parity through the public weight-processing paths."""

import copy

import pytest
import torch
from transformers import (
    CohereConfig,
    CohereForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


@pytest.mark.parametrize("logit_scale", [0.0625, 0.125, 1.0])
@pytest.mark.parametrize("use_qk_norm", [False, True])
@pytest.mark.parametrize("mode", ["raw", "process", "compatibility", "default"])
def test_cohere_processing_preserves_logits(logit_scale, use_qk_norm, mode):
    """Processing must preserve HF logits, including non-default scaling and QK norm."""
    torch.manual_seed(42)
    config = CohereConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        logit_scale=logit_scale,
        use_qk_norm=use_qk_norm,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    reference = CohereForCausalLM(config).float().eval()
    if use_qk_norm:
        with torch.no_grad():
            for layer in reference.model.layers:
                for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm):
                    norm.weight.uniform_(0.5, 1.5)
    bridge = build_bridge_from_module(
        copy.deepcopy(reference),
        "CohereForCausalLM",
        hf_config=copy.deepcopy(config),
        device="cpu",
    ).eval()
    tokens = torch.tensor([[1, 7, 3, 9, 6, 12], [1, 2, 4, 8, 5, 11]])
    options = dict(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    with torch.inference_mode():
        expected = reference(tokens, use_cache=False).logits
        for _ in range(2):
            if mode == "process":
                bridge.process_weights(**options)
            elif mode == "compatibility":
                bridge.enable_compatibility_mode(**options)
            elif mode == "default":
                bridge.enable_compatibility_mode()
            actual, cache = bridge.run_with_cache(tokens, use_cache=False)
            for layer in range(config.num_hidden_layers):
                assert f"blocks.{layer}.attn.hook_rot_q" in cache
                if use_qk_norm:
                    assert f"blocks.{layer}.attn.hook_q_normed" in cache
                    assert f"blocks.{layer}.attn.hook_k_normed" in cache
            if mode == "default":
                torch.testing.assert_close(
                    actual.log_softmax(-1), expected.log_softmax(-1), rtol=0, atol=1e-4
                )
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
                torch.testing.assert_close(bridge.embed.W_E, reference.model.embed_tokens.weight)


@pytest.mark.parametrize(
    "config_class,model_class",
    [(LlamaConfig, LlamaForCausalLM), (Qwen3Config, Qwen3ForCausalLM)],
)
def test_split_half_rotary_architectures_keep_hf_parity(config_class, model_class):
    """The shared attention changes must preserve split-half RoPE and RMS QK norms."""
    torch.manual_seed(42)
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    reference = model_class(config).float().eval()
    bridge = build_bridge_from_module(
        copy.deepcopy(reference),
        model_class.__name__,
        hf_config=copy.deepcopy(config),
        device="cpu",
    ).eval()
    tokens = torch.tensor([[1, 7, 3, 9, 6, 12], [1, 2, 4, 8, 5, 11]])
    with torch.inference_mode():
        expected = reference(tokens, use_cache=False).logits
        actual = bridge(tokens, use_cache=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-4)
