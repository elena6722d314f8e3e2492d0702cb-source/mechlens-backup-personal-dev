# Model-support release readiness

Assessment date: October 26, 2026.
Review branch: `fix/command-r-release-readiness`.
Reviewed remote revision: `440672a0d08d3929ad510a527bf89b8c30f936df`.
This report consolidates the review evidence; it does not represent a maintainer
approval or a new model-verification run. Changes remain on the review branch;
`main` has not been modified or merged into.

## Release recommendation

**BLOCKED — hold release sign-off for the proposed model-support work.**
Continue review of the implementation and documentation, but do not describe
Command-R as checkpoint-accepted or the branch as regression-clean. Close the
verification gates below before recommending shipment. Environment setup errors
are not numerical failures, but they also are not passing results.

If DeepSeek is excluded from the release scope, explicitly defer its verification
and avoid claiming newly validated V2/V3/full-R1 support. That scope decision does
not remove the Command-R checkpoint and shared-component regression gates.

## Ready / blocked / deferred summary

| State | Item | Evidence or next action |
| --- | --- | --- |
| READY for review | Command-R implementation | Cohere adapter is registered; rotary convention, optional Q/K normalization, and double logit scaling corrected. |
| READY for review | Synthetic acceptance | 24 Cohere cases plus two Llama/Qwen3 controls passed; this is not pretrained acceptance. |
| READY for review | Registry, matrix, and documentation | Implemented support and unverified checkpoint acceptance are explicitly separated; loading API and model-family mappings checked. |
| BLOCKED | Command-R checkpoint acceptance | No measured pretrained result within the strict `1e-4` threshold. Rerun the explicit slow test in a working environment. |
| BLOCKED | Cohere integration | 18 tests errored during setup; no passing integration result. |
| BLOCKED | Release regression evidence | Full `make test-pr` was interrupted; GitHub Actions reports zero branch runs as of this assessment. Obtain complete required checks on the release candidate. |
| BLOCKED if included in release sign-off | DeepSeek V2/V3/full R1 verification | Both integration files remain unrun in this review; existing comparisons do not prove independent `1e-4` checkpoint parity. |
| BLOCKED for Gemma-3 acceptance sign-off | Fresh Gemma-3 regression evidence | Historical fix is documented, but the longer-prompt checkpoint regression has not been rerun on this branch. |
| DEFERRED to triage | ActivationCache #1126 report | Display-performance complaint, no reproduction or confirmed correctness failure; not independently established as a release blocker. |
| DEFERRED to maintainers | Command-R/DeepSeek roadmap #1127 | Available record asks a question; no maintainer commitment, target release, or approval found. |

READY means the listed work or evidence is available for review, not that the
overall release is ready. DEFERRED items should remain tracked, not treated as fixed.

## Command-R implementation and scope

`CohereForCausalLM` maps to `CohereArchitectureAdapter` in TransformerBridge.
`Cohere2ForCausalLM` and `Cohere2MoeForCausalLM` are outside this adapter's scope.

The review branch corrects three numerical issues:

1. Uses Cohere's adjacent-pair rotary convention instead of split-half rotation.
2. Wires optional per-head Q/K normalization and handles its tensor shape.
3. Leaves unembedding weights unscaled because the outer HF forward applies
   `logit_scale` in raw and compatibility modes; folding it into weights scales twice.

The supported loading entry point is
`from transformer_lens.model_bridge import TransformerBridge`, followed by
`TransformerBridge.boot_transformers(model_id, device=..., dtype=...)`.
See the [Command-R guide](docs/source/content/command_r.md) for examples and limits.
An API-checked example is not evidence of a successful checkpoint load.

## Test and validation evidence

| Check | Observed result | What it establishes / does not establish |
| --- | --- | --- |
| Focused unit and registry tests, including unit `test_cohere_adapter.py` | 139 passed in a run that also had 19 setup errors | Focused tests passed; the combined run was not green. |
| `tests/acceptance/model_bridge/test_cohere_processing.py` | 26 passed: 24 Cohere cases and two controls | Synthetic HF models. Eighteen Cohere cases compare raw logits at `atol=1e-4, rtol=0`; six default-compatibility cases compare `log_softmax`. |
| `tests/acceptance/model_bridge/test_command_r.py` | One setup error; no parity measurement | Uses `trl-internal-testing/tiny-CohereForCausalLM`, independent HF/Bridge models, float32, and strict `max_abs_diff < 1e-4`. Marked slow and explicitly added to CI. |
| `tests/integration/model_bridge/test_cohere_adapter.py` | 18 setup errors | Not passed. The retry with synthetic acceptance produced 26 passes and 19 total setup errors. |
| `tests/integration/model_bridge/test_deepseek_adapter.py` | Not run; unverified | V3 synthetic fixture, not full R1 checkpoint acceptance. |
| `tests/integration/model_bridge/test_deepseek_v2_adapter.py` | Not run; unverified | Synthetic V2 and V2-Lite fixtures. |
| Registry/factory synchronization audit | 4 passed, 63 deselected | Registration consistency, not checkpoint correctness. |
| Formatting and mypy | Passed; mypy checked 300 source files | Static checks only. |
| Package build | Wheel and source distribution built during implementation review | Packaging evidence, not release acceptance. |
| Documentation | Final incremental Sphinx render succeeded with seven warnings | Not a clean full docs build. Full metadata-generating `build-docs` was stopped after model-loading errors; direct Sphinx was used. |
| Full regression / remote CI | `make test-pr` interrupted; zero Actions runs found for the branch | No complete release-candidate regression result. |

The initial checkpoint attempt encountered missing proxy support. After installing
the missing dependency, weights downloaded, but float16-to-float32 conversion
failed with `RuntimeError: Failed to initialize cpuinfo!` in the sandbox. Thus
there is **no verified tiny pretrained or full-size Command-R parity result for
the current fix**. Historical model-registry statuses do not fill this gap.

Existing DeepSeek integration comparisons use `max_diff < 0.15` against
`bridge.original_model`, not an independently loaded HF reference. Even a future
pass of those tests would not establish independent pretrained parity at `1e-4`.
There is no dedicated DeepSeek checkpoint test under `tests/acceptance`.

## Outstanding gates and required evidence

1. **Checkpoint parity:** run `uv run pytest tests/acceptance/model_bridge/test_command_r.py`
   in an environment with working CPU detection. Record candidate SHA, dependencies,
   checkpoint revision, device/dtype, command, result, and maximum logit difference.
   Keep full-size Command-R unverified unless separately tested.
2. **Cohere integration:** run all of `tests/integration/model_bridge/test_cohere_adapter.py`;
   investigate any failures beyond the current setup blocker.
3. **Shared-component regression:** complete required CI and `make test-pr` on the
   release candidate. The fixes touch shared attention/normalization behavior;
   two synthetic controls alone do not replace the broader regression suite.
4. **DeepSeek scope:** run both integration files if claiming validated V2/V3 support.
   Add or run independent checkpoint parity coverage before making a `1e-4`
   acceptance claim. Full R1 uses the V3 adapter; distillations use Llama/Qwen adapters.
5. **Gemma-3 regression protection:** locate or restore the documented longer-prompt
   and intermediate-activation regression coverage and rerun it on the candidate.
   Do not infer its pass from the historical hotfix or unit tests.
6. **Review and evidence:** retain logs in the release/CI record, resolve applicable
   docs warnings, and obtain maintainer review. Update registry/matrix statuses
   only when the corresponding evidence exists.

No individual owners or release dates have been assigned by this assessment.

## Related issues and their evidentiary limits

### Gemma-3 #1121: historical regression fixed; current rerun outstanding

The project triage note records a v2.16.0 longer-prompt reproduction at roughly
`1.3e-4`, above the `1e-4` acceptance threshold. It attributes the error to RMSNorm
epsilon placement, records strengthened acceptance coverage and a corrected
maximum difference of `2.9e-5`, and says v2.16.1 shipped on June 27, 2026.
The July 7 maintainer update reports clean post-release monitoring.

This supports a resolved historical defect, not a confirmed active regression.
Current normalization code places epsilon inside the square-root/rsqrt operation,
consistent with the fix, but no fresh end-to-end Gemma-3 result was established in
this review. Legacy HookedTransformer acceptance is also quarantined in the current
repository; see [test quarantines](tests/QUARANTINES.md).

### ActivationCache #1126: unresolved performance report, not a confirmed blocker

The available July 8 notification reports that printing a large cache hangs a
notebook and requests truncation. No reproducer, timings, maintainer decision, or
verified fix was found. Current `ActivationCache.__repr__` formats all cache keys,
not tensor contents. Large output is plausible with many entries, but the hang
has not been reproduced here. Keep it in triage; escalate if ordinary release
workflows reproduce a severe hang. There is no evidence in the report of incorrect
activations or model outputs.

### Roadmap #1127: question, not a decision

The July 10 notification asks whether Command-R (#1124) and a DeepSeek adapter
are on the near-term roadmap. The available thread contains no maintainer reply,
commitment, priority, or target release. Adapter implementation does not establish
a roadmap decision or approval to ship.

**Issue-number mismatch:** live upstream #1126 is a closed “Attention cleanup” PR,
and #1127 is a closed “Benchmarking cross comparison revision” PR. Those live
records are different from the project notifications described above. Their closed
status must not be used to claim these project reports have been resolved.

## Review artifacts and provenance

- [Adapter registry](Configs/adapter_registry.json): Command-R implemented,
  checkpoint acceptance unverified; existing entries preserved.
- [Support matrix](Adapters/adapter_support_matrix.xlsx): matches the updated Drive
  workbook byte for byte at publication; resides under `Adapters`, not `Configs`.
- [Command-R guide](docs/source/content/command_r.md) and
  [DeepSeek guide](docs/source/content/deepseek.md): loading, architecture scope,
  and validation limits. Contributor guidance now avoids double logit scaling.
- Local review logs: `focused.log`, `checkpoint.log`, `cohere-final.log`,
  `registry-audit.log`, `registry-test-pr.log`, and documentation-check logs.
  These session logs are not committed artifacts; archive the candidate's fresh
  logs in CI before release rather than relying on this summary alone.
- Project records consulted: `triage_1121_gemma3_logits.md`, the v2.16.1 release
  notification, July 7 maintainer update, and July 8/10 issue notifications.
  Their historical claims are distinguished above from current repository checks.

Acceptance results in this document are evidence from the review session, not a
claim that every test has been rerun at the documentation-only revision listed
at the top.
