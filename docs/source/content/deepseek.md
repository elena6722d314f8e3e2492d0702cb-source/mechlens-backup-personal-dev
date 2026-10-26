# DeepSeek support

TransformerBridge selects adapters by the checkpoint's Hugging Face architecture,
not by the DeepSeek name. V2 and V3 are implemented; full R1 shares the V3 adapter.

| Model family | HF architecture | Bridge adapter |
| --- | --- | --- |
| V2, V2-Lite, Coder-V2 | `DeepseekV2ForCausalLM` | `DeepSeekV2ArchitectureAdapter` |
| V3 and full R1 (including R1-Zero and R1-0528 in the registry) | `DeepseekV3ForCausalLM` | `DeepSeekV3ArchitectureAdapter` |
| R1-Distill-Llama | `LlamaForCausalLM` | `LlamaArchitectureAdapter` |
| R1-Distill-Qwen | `Qwen2ForCausalLM` | `Qwen2ArchitectureAdapter` |
| R1-0528-Qwen3-8B | `Qwen3ForCausalLM` | `Qwen3ArchitectureAdapter` |

The V2 adapter handles both compressed Q projections and the direct Q projection
used when `q_lora_rank=None` (V2-Lite). V2 and V3 use Multi-Head Latent Attention
(MLA), dense initial layers, and MoE layers with shared experts. Their MLA blocks
do not expose the legacy split-Q/K/V input aliases or pre-ln1 capture described
in [Compatibility Mode](compatibility_mode.md). These restrictions do not apply
to R1 distillations merely because they carry the R1 name; their underlying
Llama/Qwen adapter determines hook behavior.

## Verification status

Implemented architecture support does not establish checkpoint acceptance or
release readiness. In this review, neither
`tests/integration/model_bridge/test_deepseek_v2_adapter.py` nor
`tests/integration/model_bridge/test_deepseek_adapter.py` has a verified run.
The first covers synthetic V2 and V2-Lite configurations; the second covers a
synthetic V3 configuration. They save randomly initialized models locally and
load a GPT-2 tokenizer. Their current logit comparisons use `max_diff < 0.15`
against `bridge.original_model`, rather than an independently loaded reference.
They therefore do not establish independent pretrained-checkpoint parity within
`1e-4`, even if they pass. There is no dedicated DeepSeek checkpoint test under
`tests/acceptance` in the current repository.

The model registry contains historical per-checkpoint results, which must be read
with their notes. Full R1, R1-Zero, and R1-0528 are recorded as skipped for memory
limits. Some distillations have verification records, but others have only
architecture validation, and the Qwen 14B record notes a weight-modification
failure. A distillation's result does not verify the full R1 architecture or the
current review branch. No new DeepSeek checkpoint parity result is claimed here.

## Loading

Use `TransformerBridge.boot_transformers` with an exact checkpoint ID whose
architecture appears above. This example shows the API for a Qwen2 distillation,
not full R1 or a new verified loading result:

```python
import torch
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device="cpu",
    dtype=torch.float32,
)
```

Full V2/V3/R1 checkpoints have much larger memory requirements. Choose hardware,
precision, and checkpoint access accordingly; `device_map="auto"` can be used
instead of `device` for dispatched inference. Loading a quantized variant or a
new architecture is not guaranteed by a family name. Reduced-precision inference
is not float32 acceptance evidence.

Run the existing integration coverage explicitly from the repository root:

```bash
uv run pytest tests/integration/model_bridge/test_deepseek_v2_adapter.py
uv run pytest tests/integration/model_bridge/test_deepseek_adapter.py
```

Record actual results before changing their review status from unverified.
