# Command-R

## Support status

Command-R is implemented in `TransformerBridge` through
`CohereArchitectureAdapter` for Hugging Face's `CohereForCausalLM` architecture.
This does not include `Cohere2ForCausalLM` or `Cohere2MoeForCausalLM`.

On `fix/command-r-release-readiness`, 24 synthetic Cohere acceptance cases passed:
18 compare raw logits at `atol=1e-4, rtol=0`, and six compare `log_softmax` after
default compatibility processing. These tests use randomly initialized HF models,
not pretrained checkpoints.

**Pretrained-checkpoint acceptance remains unverified.** The tiny-checkpoint test
requires the maximum absolute float32 logit difference from an independent HF
reference to be strictly less than `1e-4`. Its last attempt stopped during model
setup with `RuntimeError: Failed to initialize cpuinfo!` in the sandbox, before
any parity measurement. Neither that checkpoint nor a full-size Command-R model
has a verified parity result for the current fix. Implemented support is not a
release-readiness sign-off.

## Loading

Use the Bridge API from this review branch. The tiny model is an architecture
test fixture, not a useful conversational model:

```python
import torch
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "trl-internal-testing/tiny-CohereForCausalLM",
    device="cpu",
    dtype=torch.float32,
)
bridge.eval()
tokens = torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
with torch.inference_mode():
    logits = bridge(tokens, use_cache=False)
```

This is the current loading API, not a claim that the checkpoint example has
completed successfully in the sandbox. For a full-size model, replace the model
ID with a checkpoint whose HF architecture is `CohereForCausalLM`, for example
`CohereLabs/c4ai-command-r-v01`. Obtain access to gated checkpoints and configure
`HF_TOKEN` as described in [Getting Started](getting_started.md#hf_token).
Full-size models need substantially more memory than the tiny fixture; select
`device` and `dtype` for your hardware. The loader also accepts `device_map="auto"`
instead of `device` for dispatched inference. Reduced precision is not the
float32 acceptance configuration.

Raw HF weights are the default. For parity comparisons, keep compatibility mode
disabled and load the independent HF reference with `dtype=torch.float32` and
`attn_implementation="eager"`; Bridge uses eager attention. Compare identical token
IDs so tokenizer/BOS behavior does not confound the result.

The adapter uses adjacent-pair rotary embeddings and optional per-head Q/K
normalization. Leave unembedding weights unscaled: the outer HF forward applies
`logit_scale` once in both raw and compatibility modes. Default compatibility
processing can center logits, so its synthetic tests compare `log_softmax` rather
than requiring identical raw logits.

## Acceptance commands

From the repository root:

```bash
uv run pytest tests/acceptance/model_bridge/test_cohere_processing.py
# Requires checkpoint access and a working PyTorch CPU environment:
uv run pytest tests/acceptance/model_bridge/test_command_r.py
```

The first file includes 24 Cohere cases and two other-architecture controls.
The second is marked `slow` and must be run explicitly when the general suite
excludes slow tests. Record a successful checkpoint result before changing its
acceptance status from unverified.
