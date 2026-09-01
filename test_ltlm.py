#!/usr/bin/env python
"""CPU checks for the LTLM adaptation (tiny GPT-2, no Hub download).

The silent failure modes this guards, because both look like a healthy run:

  * thoughts not actually reaching the packed window (zero-init is *supposed*
    to do that at step 0; after we break the zeros, different ``z`` must
    change logits);
  * packed position ids / sequence length shifting (they must stay 0..1019);
  * prefix posterior accidentally seeing body score tokens;
  * diffusion cross-attn ignoring the performance.

Run on a compute node (login-node Python is SIGKILL'd around 2–3 GB RSS)::

    srun -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:1 \
        --cpus-per-task=4 --mem=64G --time=1:00:00 \
        bash -lc 'source /home/wjl86/miniconda3/etc/profile.d/conda.sh && conda activate base && python -u test_ltlm.py'
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME
from anticipation.packed_sequence import ALTERNATING_START, dummy_rest_triplet
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
    VOCAB_SIZE,
)
from ltlm import (
    LTLMConfig,
    attach_latent_thoughts,
    infer_posterior,
    ltlm_autoregressive_generate_score,
    ltlm_forward,
    ones_attention_mask,
    prefix_labels,
    reshape_z,
    z_length,
)
from ltlm_diffusion import DiffusionConfig, PlanDiffusion
from plan_vq import BODY_SLOTS, NUM_PERF_NOTES, PACKED_LENGTH, split_packed_batch

failures = []

PERF_SPACING = (MAX_TIME - 20) // NUM_PERF_NOTES
SCORE_SPACING = (MAX_TIME - 3) // BODY_SLOTS


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def make_packed_window(seed):
    rng = torch.Generator().manual_seed(seed)

    def rand(high):
        return int(torch.randint(0, high, (1,), generator=rng).item())

    perf = [
        (PERF_SPACING * i + rand(20), 20 + rand(40), 21 + rand(80))
        for i in range(NUM_PERF_NOTES)
    ]
    score = [
        (SCORE_SPACING * i + rand(3), 25 + rand(50), 21 + rand(80))
        for i in range(BODY_SLOTS)
    ]
    tokens = []
    for i in range(ALTERNATING_START // 6):
        onset, duration, pitch = perf[i]
        tokens += [ATIME_OFFSET + onset, ADUR_OFFSET + duration, ANOTE_OFFSET + pitch]
        tokens += dummy_rest_triplet(0)
    for slot in range(BODY_SLOTS):
        onset, duration, pitch = score[slot]
        tokens += [TIME_OFFSET + onset, DUR_OFFSET + duration, NOTE_OFFSET + pitch]
        onset, duration, pitch = perf[ALTERNATING_START // 6 + slot]
        tokens += [ATIME_OFFSET + onset, ADUR_OFFSET + duration, ANOTE_OFFSET + pitch]
    assert len(tokens) == PACKED_LENGTH
    return torch.tensor(tokens, dtype=torch.long)


def tiny_lm():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE, n_positions=1024, n_embd=32, n_layer=2, n_head=4,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def break_zero_init(model):
    """The production out-proj is zero-init; tests that need a visible z undo it."""
    for block in model.transformer.h:
        nn.init.normal_(block.thought_cross_attn.out.weight, std=0.05)
        if block.thought_cross_attn.out.bias is not None:
            nn.init.zeros_(block.thought_cross_attn.out.bias)


# ---------------------------------------------------------------------------
print("packed window helpers")
packed = torch.stack([make_packed_window(1), make_packed_window(2)])
check("window length", packed.shape == (2, PACKED_LENGTH), str(tuple(packed.shape)))
perf, score, valid = split_packed_batch(packed)
check("split_packed_batch perf", tuple(perf.shape) == (2, NUM_PERF_NOTES, 3), str(perf.shape))
check("split_packed_batch score", tuple(score.shape) == (2, BODY_SLOTS, 3), str(score.shape))

labels = packed.clone()
pref = prefix_labels(labels, ALTERNATING_START)
check("prefix keeps the lookahead", bool((pref[:, :ALTERNATING_START] == labels[:, :ALTERNATING_START]).all()))
check("prefix masks the body", bool((pref[:, ALTERNATING_START:] == -100).all()))
check("prefix length is ALTERNATING_START=192", ALTERNATING_START == 192)

# The variational likelihood covers only what the model generates: the real
# score triplets. Controls are conditioning and the prefix's dummy rests are
# placeholders, so both are -100.
from packed_dataset import TokenizedDataset  # noqa: E402
from ltlm import score_only_labels  # noqa: E402

score_token_mask = torch.stack([
    TokenizedDataset._build_score_token_mask(None, row) for row in packed
])
score_only = score_only_labels(labels, score_token_mask)
check("score-only labels keep every real score triplet",
      bool((score_only[score_token_mask] == labels[score_token_mask]).all()))
check("score-only labels drop the controls and the dummy rests",
      bool((score_only[~score_token_mask] == -100).all()))
kept = int(score_token_mask[0].sum())
check("score-only keeps 138 slots x 3 tokens", kept == BODY_SLOTS * 3, str(kept))

# The prefix arm must survive --loss_mask_performance_tokens: it builds its
# observation from input_ids, so masking the controls out of the decoder's loss
# cannot silently reduce it to "no plan".
from ltlm import prefix_observation_labels  # noqa: E402

controls_masked = labels.clone()
controls_masked[~score_token_mask] = -100
obs = prefix_observation_labels(packed, ALTERNATING_START)
check("prefix observation ignores the training label mask",
      bool((obs[:, :ALTERNATING_START] == packed[:, :ALTERNATING_START]).all()))
check("prefix observation still stops at ALTERNATING_START",
      bool((obs[:, ALTERNATING_START:] == -100).all()))
check("prefix observation keeps the 32 lookahead controls that the loss mask drops",
      int((obs[0, :ALTERNATING_START] != -100).sum())
      > int((prefix_labels(controls_masked)[0, :ALTERNATING_START] != -100).sum()))

def last_logits(model, tokens, z=None, **kwargs):
    """Keep only the last-position slice so tests do not hold [B, 1020, 55k] tensors."""
    if z is None:
        out = model(input_ids=tokens, attention_mask=ones_attention_mask(tokens), **kwargs)
    else:
        out = ltlm_forward(
            model, tokens, z, attention_mask=ones_attention_mask(tokens), **kwargs
        )
    logits = out.logits[:, -1, :].detach()
    seq_len = out.logits.shape[1]
    del out
    return logits, seq_len


# ---------------------------------------------------------------------------
print("\nz injection (sequence length and zero-init)")
cfg = LTLMConfig(thoughts_per_layer=2, z_dim=32, mcmc_steps=2, dropout=0.0)
model = tiny_lm()
single = packed[:1]
single_labels = labels[:1]
with torch.no_grad():
    vanilla, _ = last_logits(model, single)
attach_latent_thoughts(model, cfg, verbose=False)
n_z = z_length(model, cfg)
check("z length = layers * thoughts_per_layer", n_z == 4, str(n_z))

z_a = torch.randn(1, n_z, 32)
z_b = torch.randn(1, n_z, 32)
with torch.no_grad():
    zero_init_with_z, seq_len = last_logits(model, single, z_a)
    zero_init_other_z, _ = last_logits(model, single, z_b)
check("forward keeps packed length", seq_len == PACKED_LENGTH, str(seq_len))
check(
    "zero-init cross-attn leaves logits identical to vanilla GPT-2",
    torch.allclose(vanilla, zero_init_with_z, atol=1e-5, rtol=1e-5),
    f"max abs {float((vanilla - zero_init_with_z).abs().max())}",
)
check(
    "zero-init: two different z still match (thoughts start invisible)",
    torch.allclose(zero_init_with_z, zero_init_other_z, atol=1e-5, rtol=1e-5),
)
del vanilla, zero_init_with_z, zero_init_other_z

break_zero_init(model)
with torch.no_grad():
    logits_a, _ = last_logits(model, single, z_a)
    logits_b, _ = last_logits(model, single, z_b)
    logits_a_again, _ = last_logits(model, single, z_a)
check(
    "after breaking zero-init, z changes logits",
    float((logits_a - logits_b).abs().max()) > 1e-4,
    f"max abs {float((logits_a - logits_b).abs().max())}",
)
check("same z is deterministic in eval", torch.allclose(logits_a, logits_a_again, atol=1e-5))

pos = torch.arange(PACKED_LENGTH).unsqueeze(0)
with torch.no_grad():
    logits_pos, _ = last_logits(model, single, z_a, position_ids=pos)
check(
    "explicit position ids 0..1019 agree with the default",
    torch.allclose(logits_a, logits_pos, atol=1e-5),
)
del logits_a, logits_b, logits_a_again, logits_pos

# ---------------------------------------------------------------------------
print("\nMCMC / AdamVI shapes")
model.eval()
z, mu, log_var, stats = infer_posterior(
    model, single, single_labels, cfg,
    attention_mask=ones_attention_mask(single),
    steps=2, fast_lr=0.05,
)
check("inferred z shape", tuple(z.shape) == (1, n_z, 32), str(tuple(z.shape)))
check("mu shape matches z", tuple(mu.shape) == tuple(z.shape))
check("log_var shape matches z", tuple(log_var.shape) == tuple(z.shape))
check("nll is finite", stats["nll"] == stats["nll"] and abs(stats["nll"]) < 1e6, str(stats["nll"]))
check("kl is finite", stats["kl"] == stats["kl"], str(stats["kl"]))
check("reshape_z layers", tuple(reshape_z(z, 2, 2).shape) == (1, 2, 2, 32))

z_pref, mu_pref, _, stats_pref = infer_posterior(
    model, single, prefix_labels(single_labels), cfg,
    attention_mask=ones_attention_mask(single),
    steps=2, fast_lr=0.05,
)
check("prefix posterior also returns the full z shape", tuple(z_pref.shape) == tuple(z.shape))
check(
    "prefix and oracle posteriors differ once z is visible",
    float((mu - mu_pref).abs().max()) > 0.0,
    f"max abs {float((mu - mu_pref).abs().max())}",
)

print("\nLangevin shapes")
cfg_l = LTLMConfig(thoughts_per_layer=2, z_dim=32, mcmc_steps=2, inference_method="langevin",
                   langevin_step_size=0.01, dropout=0.0)
z_l, mu_l, _, stats_l = infer_posterior(
    model, single, single_labels, cfg_l,
    attention_mask=ones_attention_mask(single), steps=2,
)
check("langevin z shape", tuple(z_l.shape) == (1, n_z, 32), str(tuple(z_l.shape)))
check("langevin nll finite", stats_l["nll"] == stats_l["nll"])

# ---------------------------------------------------------------------------
print("\nAR decode preserves controls and length")
generated = ltlm_autoregressive_generate_score(
    model, single, z, device="cpu", constrain_score_tokens=True,
    ground_truth_score_tokens_to_feed=0,
)
check("AR output shape", tuple(generated.shape) == tuple(single.shape), str(tuple(generated.shape)))
check(
    "AR copies the packed prefix through",
    bool((generated[:, :ALTERNATING_START] == single[:, :ALTERNATING_START]).all()),
)
control_cols = []
for pos in range(ALTERNATING_START, PACKED_LENGTH, 6):
    control_cols.extend([pos + 3, pos + 4, pos + 5])
control_cols = [c for c in control_cols if c < PACKED_LENGTH]
check(
    "AR teacher-forces body controls",
    bool((generated[:, control_cols] == single[:, control_cols]).all()),
)

# The KV-cached step must see the whole cache. transformers reads
# attention_mask as a padding mask over past+current keys, so a chunk-shaped
# all-ones mask silently truncates the context to the current token and the
# rollout still comes out the right shape.
prefix_ids = single[:, :ALTERNATING_START]
next_id = single[:, ALTERNATING_START:ALTERNATING_START + 1]
with torch.no_grad():
    primed = ltlm_forward(model, prefix_ids, z, use_cache=True)
    cached = ltlm_forward(
        model, next_id, z, past_key_values=primed.past_key_values, use_cache=True
    )
    uncached = ltlm_forward(
        model, single[:, :ALTERNATING_START + 1], z,
        attention_mask=ones_attention_mask(single[:, :ALTERNATING_START + 1]),
    )
cache_gap = float((cached.logits[:, -1, :] - uncached.logits[:, -1, :]).abs().max())
check(
    "cached decode step matches the uncached forward (cache is not truncated)",
    cache_gap < 1e-3,
    f"max abs {cache_gap}",
)

# infer_posterior optimizes the ELBO, so it needs a graph even though
# evaluate_ltlm() wraps the whole validation pass in @torch.no_grad().
@torch.no_grad()
def _posterior_under_no_grad():
    return infer_posterior(
        model, single, single_labels, cfg,
        attention_mask=ones_attention_mask(single), steps=2, fast_lr=0.05,
    )


try:
    _, mu_ng, _, _ = _posterior_under_no_grad()
    check(
        "posterior inference works inside torch.no_grad (validation path)",
        float(mu_ng.abs().max()) > 0.0,
        "mu never left the prior",
    )
except RuntimeError as error:
    check("posterior inference works inside torch.no_grad (validation path)",
          False, str(error))

# ---------------------------------------------------------------------------
print("\ndiffusion cross-attn")
diff = PlanDiffusion(DiffusionConfig(
    z_dim=32, n_thoughts=n_z, d_model=32, n_head=4, n_layer=1,
    perf_layers=1, fourier_bands=4, dropout=0.0, timesteps=10, sample_steps=3,
))
diff.train()
loss = diff.loss(mu.detach(), single)
check("diffusion loss finite", bool(torch.isfinite(loss.detach())), str(float(loss.detach())))
loss.backward()
has_perf_grad = diff.performance.pitch.weight.grad is not None and bool(
    diff.performance.pitch.weight.grad.abs().sum() > 0
)
check("diffusion loss reaches the performance encoder", has_perf_grad)

check(
    "diffusion forward() routes to loss (DDP-visible entry point)",
    torch.isfinite(diff(mu.detach(), single).detach()),
)

# The oracle posterior mean starts at 0 and its scale is unknown a priori, so
# the target is normalised by a running RMS; sampling must undo that.
scaled = PlanDiffusion(DiffusionConfig(
    z_dim=32, n_thoughts=n_z, d_model=32, n_head=4, n_layer=1,
    perf_layers=1, fourier_bands=4, dropout=0.0, timesteps=10, sample_steps=3,
))
scaled.train()
big_target = torch.randn(1, n_z, 32) * 25.0
scaled.loss(big_target, single)
check(
    "diffusion tracks the target's RMS",
    abs(float(scaled.target_rms) - float(big_target.pow(2).mean().sqrt())) < 1e-3,
    f"target_rms={float(scaled.target_rms):.4f}",
)
scaled.eval()
big_sample = scaled.sample(single, steps=3)
check(
    "sampling restores the target scale (not unit-variance latents)",
    float(big_sample.abs().mean()) > 1.0,
    f"mean abs {float(big_sample.abs().mean()):.4f}",
)

# The denoiser models mu; sample() re-adds q's spread so the LM still sees a
# z-shaped object. The bare mu_hat is what the fidelity metrics compare.
diff.posterior_sigma.fill_(0.75)
torch.manual_seed(3)
bare = diff.sample(single, steps=3, add_posterior_noise=False)
torch.manual_seed(3)
noised = diff.sample(single, steps=3, add_posterior_noise=True)
check("add_posterior_noise=False returns the bare mu_hat",
      not torch.allclose(bare, noised, atol=1e-6))
gap = (noised - bare).std()
check("re-added noise carries the tracked posterior sigma",
      abs(float(gap) - 0.75) < 0.15, f"std {float(gap):.3f} vs sigma 0.75")

log_var = torch.full((1, n_z, 32), -2.0)
before = float(diff.posterior_sigma)
diff.update_posterior_sigma(log_var)
check("update_posterior_sigma tracks exp(0.5*log_var)",
      float(diff.posterior_sigma) < before,
      f"{before:.3f} -> {float(diff.posterior_sigma):.3f} (target {float((0.5*log_var).exp().mean()):.3f})")

diff.eval()
sampled = diff.sample(single, steps=3)
check("diffusion sample shape", tuple(sampled.shape) == (1, n_z, 32), str(tuple(sampled.shape)))

with torch.no_grad():
    # Same noisy z, two different performances: the denoiser must move.
    t = torch.zeros(2, dtype=torch.long)
    z_pair = mu.detach().expand(2, -1, -1).contiguous()
    noise = torch.randn_like(z_pair)
    z_t, _ = diff.q_sample(z_pair.float(), t, noise=noise)
    perf_a = diff.encode_performance(packed)
    perf_b = diff.encode_performance(packed.roll(1, dims=0))
    eps_a = diff._denoise(z_t.to(perf_a.dtype), t, perf_a)
    eps_b = diff._denoise(z_t.to(perf_b.dtype), t, perf_b)
shift = float((eps_a - eps_b).abs().max())
check("denoiser reads the performance (cross-attn is not a stub)", shift > 1e-5, f"max abs {shift}")

# ---------------------------------------------------------------------------
print("\npiano-roll figure (VQ helper reused on packed notes)")
from plan_vq_viz import piano_roll_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

perf, score, valid = split_packed_batch(packed)
# Identical pred -> all green; one pitch flip -> that slot red.
predicted = score.clone()
predicted[0, 0, 2] = (predicted[0, 0, 2] + 7) % MAX_PITCH
fig = piano_roll_figure(
    perf, score, valid, predicted,
    step=0, max_examples=2,
    suptitle="LTLM AR decode (oracle)",
    pred_heading="generated score (oracle)",
    window_ids=[0, 1],
)
check("piano-roll figure is 3 columns", fig.axes is not None and len(fig.axes) == 6, str(len(fig.axes)))
suptitle = fig.get_suptitle() if hasattr(fig, "get_suptitle") else (
    fig._suptitle.get_text() if getattr(fig, "_suptitle", None) is not None else ""
)
check("piano-roll titles name the mode and window",
      "oracle" in suptitle and any("window 0" in (ax.get_title() or "") for ax in fig.axes),
      suptitle)
plt.close(fig)

# ---------------------------------------------------------------------------
print("\nmetric wiring (names the trainer logs)")
expected = []
for mode in ("oracle", "prefix", "diffusion"):
    expected.extend([
        f"val/{mode}/loss",
        f"val/{mode}/ar_pitch_accuracy",
        f"val/{mode}/ar_onset_accuracy",
        f"val/{mode}/ar_duration_accuracy",
        f"val/{mode}/piano_roll",
    ])
from train_ltlm import MODES, _fixed_piano_roll_indices  # noqa: E402
check("train_ltlm.MODES is oracle/prefix/diffusion", tuple(MODES) == ("oracle", "prefix", "diffusion"))
check("expected wandb names include piano rolls", len(expected) == 15)

class _FakeVal:
    def __len__(self):
        return 10

check("fixed windows are the first N of the val file",
      _fixed_piano_roll_indices(_FakeVal(), 3) == [0, 1, 2])
check("--piano_roll_examples 0 disables rolls",
      _fixed_piano_roll_indices(_FakeVal(), 0) == [])

print("\n" + ("All checks passed." if not failures else f"{len(failures)} failure(s): {failures}"))
raise SystemExit(1 if failures else 0)
