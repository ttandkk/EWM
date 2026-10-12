# EWM v1 Architecture Specification

## Frozen design

EWM v1 replaces OpenWAM's Video Expert with a 4-bin Event Expert while
preserving OpenWAM joint_self_attn / MoT interaction.

    3D Event Stem
    -> OpenWAM-style full Event/Action mixed self-attention
    -> 3D Event Decoder

DINO RGB and T5 language are external context. They are not mixed
Event/Action self-attention tokens.

## Dataset contract and canonical voxel layout

The loader provides:

    obs_video:     uint8    [B, 2 camera, 3, 256, 256]
    obs_voxel:     float32  [B, 2 camera, 8, 256, 256]
    proprio:       float32  [B, 8]
    action:        float32  [B, K, 7]
    future_voxel:  float32  [B, K, 2 camera, 8, 256, 256]

Voxel values are normalized counts in [0,1]. Convert the fixed p-major
channel layout to the canonical event layout before Event3DEncoder:

    channels 0..3: negative polarity, bins 0..3
    channels 4..7: positive polarity, bins 0..3

    [B, K, camera, 8, H, W]
    -> [B, K, camera, polarity=2, bin=4, H, W]

Polarity remains an explicit axis; count values always remain nonnegative.

## Exact dynamics alignment

Definitions:

    V_k     = events in (t_{k-1}, t_k]
    a_k     = action applied from t_k to t_{k+1}
    V_{k+1}= events in (t_k, t_{k+1}]

The EWM sample contract is:

    clean observations:  V_k, s_k
    action targets:      a_k, ..., a_{k+K-1}
    event targets:       V_{k+1}, ..., V_{k+K}

RoPE time is aligned to transitions, not array offset:

    coordinate 0: Event V_k, Action state s_k
    coordinate i: Event V_{k+i}, Action a_{k+i-1}, i=1..K

Thus a_k and V_{k+1} both receive time coordinate 1. This is the mandatory
off-by-one invariant.

## Lightweight 3D Event Stem

Process each camera and action interval independently, with B, K, camera
merged only as Conv3D batch dimension:

    [B, K, 2, 2, 4, 256, 256]
    -> [B*K*2, 2, 4, 256, 256]
    -> Event3DEncoder
    -> [B*K*2, D, 1, 16, 16]
    -> [B, K, 2, 256, D]

Use D=512 in v1. The 3D stem explicitly compresses four high-frequency event
bins to one 30 Hz latent interval and compresses spatial resolution by 16.

Recommended encoder geometry:

    input                              [2,   4, 256, 256]
    ResBlock3D + downsample (1,2,2)    [C1,  4, 128, 128]
    ResBlock3D + downsample (1,2,2)    [C2,  4,  64,  64]
    ResBlock3D + downsample (2,2,2)    [C3,  2,  32,  32]
    ResBlock3D + downsample (2,2,2)    [D,   1,  16,  16]

Use lightweight GroupNorm plus SiLU/GELU residual blocks. This is not an Event
VAE: there is no KL term, pretraining stage, cached latent dataset, or
latent-space flow objective.

## Event and Action streams

The Event stream is:

    E = [Enc(V_k clean), Enc(V_t,k+1), ..., Enc(V_t,k+K)]

The Action stream is:

    A = [MLP(s_k clean), MLP(a_t,k), ..., MLP(a_t,k+K-1)]

Only future hidden tokens are decoded. Clean prefix tokens are conditions
inside their own stream. v1 needs no clean/noisy role embedding, segment
embedding, special attention mask, or causal mask because all layouts are
fixed.

## Shared multimodal RoPE

All Event and Action Q/K tensors use the same rotary head geometry. For
head_dim=64:

    time subspace: 16 dimensions
    y subspace:    24 dimensions
    x subspace:    24 dimensions

Event tokens apply 3D RoPE(t,y,x), where t is in 0..K and y,x are in 0..15.

Action and state tokens apply only time RoPE in the shared time subspace;
their y/x subspaces are identity. This common basis is required for valid
Event-Action mixed attention. Add learned cam0/cam_wrist embeddings to Event
tokens. Do not add learned spatial, future-step, role, or segment embeddings.

## External RGB and language context

RGB and language are separate external KV context.

    RGB [B,2,3,H,W]
    -> reshape [2B,3,H,W]
    -> frozen DINO
    -> restore camera dimension
    -> project to D
    -> add RGB modality embedding and camera embedding

    prompt strings
    -> frozen T5SmallTextEncoder
    -> [B,L,512]
    -> project to D when needed
    -> add text modality embedding

    C = [C_rgb_cam0 ; C_rgb_wrist ; C_text]

Never hard-code the DINO token count; infer it from the actual frozen DINO
output. Modality embeddings are retained to distinguish RGB context from text
context. They are unrelated to omitted clean/noisy role embeddings.

Implementation requirement: batch RGB preprocessing as tensor operations on the
DINO device/dtype (uint8 -> float / 255 -> the existing OpenWAM DINO ImageNet
mean/std). Do not route training RGB through CPU, NumPy or PIL. Frozen DINO/T5
stay in eval mode when the enclosing model trains. Text padding remains masked.

## OpenWAM-style full mixed self-attention

Every v1 joint layer uses strict full mixed Event/Action self-attention with
mutual visibility. Each branch has independent AdaLN, QKV projections, output
projection, and FFN:

    K_all = [K_E ; K_A]
    V_all = [V_E ; V_A]

    O_E = Attn(Q_E, K_all, V_all)
    O_A = Attn(Q_A, K_all, V_all)

The implementation may concatenate Q_E/Q_A, call SDPA once on the common K/V,
and split the result; this is mathematically equivalent to the two expressions
above and does not change the independent expert parameters.

The visibility matrix contains all blocks:

    [ E -> E    E -> A ]
    [ A -> E    A -> A ]

There is no separate Event-to-Action cross-attention and no additional joint
attention block. Mixed self-attention already implements the complete MoT
interaction.

After the mixed residual, each stream independently cross-attends to external
context C, then applies its own FFN:

    Event mixed hidden  -> CrossAttn(Q=E, KV=C) -> Event FFN
    Action mixed hidden -> CrossAttn(Q=A, KV=C) -> Action FFN

Event and Action use separate AdaLN time MLPs but share one scalar flow time.

## Rectified flow

Flow is defined directly in normalized voxel space and normalized action
space, not in learned Event3DEncoder representation.

For shared t sampled uniformly from [0,1]:

    eps_E = Gaussian noise matching future_voxel
    eps_A = Gaussian noise matching future_action

    X_t,E = (1-t) X_0,E + t eps_E
    X_t,A = (1-t) X_0,A + t eps_A

    v_target,E = eps_E - X_0,E
    v_target,A = eps_A - X_0,A

The Event decoder predicts:

    v_hat_E: [B,K,2 camera,2 polarity,4 bin,256,256]

The Action head predicts:

    v_hat_A: [B,K,7]

Use independently mean-reduced losses:

    L_event  = mean((v_hat_E - v_target,E)^2)
    L_action = mean((v_hat_A - v_target,A)^2)
    L_total  = L_event + L_action

Monitor active-region and inactive-region velocity MSE plus active fraction,
where the mask is clean future voxel > 0. These are detached diagnostics; they
do not reweight or replace either global mean loss. Empty regions report zero
and must be interpreted with their occupancy. Clean prefixes receive time AdaLN
modulation, but their input values are never noised.

At inference initialize both future variables as Gaussian noise at t=1 and
integrate jointly backward:

    X_{t-dt} = X_t - dt * v_theta(X_t, t, condition)

After sampling, clamp event prediction to [0,1]. Clamp action to [-1,1], then
call EWMNormalizer.denormalize_action. For event visualization only, call
EWMNormalizer.denormalize_voxel; keep continuous counts by default and use
round_counts=True only for approximate discrete event counts.

## Lightweight 3D Event Decoder

Decode only future Event hidden tokens:

    [B,K,2,256,D]
    -> [B*K*2,D,1,16,16]
    -> Event3DDecoder
    -> [B,K,2,2,4,256,256]

Recommended non-causal decoder geometry:

    [D,  1,  16,  16]
    -> ResBlock3D -> upsample (2,2,2) -> [C1,2, 32, 32]
    -> ResBlock3D -> upsample (2,2,2) -> [C2,4, 64, 64]
    -> ResBlock3D -> upsample (1,2,2) -> [C3,4,128,128]
    -> ResBlock3D -> upsample (1,2,2) -> [C4,4,256,256]
    -> Conv3D -> [2 polarity,4 bin,256,256]

Decoder output is unbounded voxel-space velocity. Do not use sigmoid, output
clamping, causal convolution, or video-cache logic inside the decoder.

The current decoder implements each stage as ConvTranspose3D upsampling
(kernel_size == stride), followed by two residual Conv3D refinements. The
geometry above is mandatory; the shown ResBlock/upsample ordering is a
recommendation. Interpolation + Conv3D is a later artifact-reduction ablation,
not a claimed feature of the current implementation. Inspect actual event
edges, background grid artifacts and four-bin motion in reconstruction tests.

## Model scale and K schedule

v1 defaults:

    D=512
    heads=8
    approximately 8 joint layers
    full mixed attention in every joint layer
    latent event grid=16 by 16
    local temporal compression=4 to 1

K=8 is bring-up only:

    Event tokens  = (8+1)*2*256 = 4608
    Action tokens = 9

The implementation must support arbitrary K. Experimental order:

    K=8  -> validate end-to-end training and sampling
    K=16 -> measure quality, memory, throughput
    K=32 -> target policy horizon

For K=32, Event tokens are (32+1)*2*256=16896 and Action tokens are 33.
The dominant cost is Event-to-Event attention, not Action tokens. Preserve
full OpenWAM-style mixed attention first. If needed, try FlashAttention,
activation checkpointing, and sequence parallel before changing architecture.
Factorized/window Event attention is a later efficiency ablation, not v1.

Codec microbatching is an engineering memory/throughput setting, not an
attention factorization. Start GPU profiling at codec_batch_size=16; compare
16/32/64 on the target hardware. CPU diagnostics explicitly use 2. No GPU
throughput or memory claim follows from small-model CPU tests.

## Required implementation gates

1. Event3DEncoder and Event3DDecoder:
   shape, finite-value, backward-gradient, and axis-separation tests.
   Run a small reconstruction sanity experiment for 4-to-1 temporal and
   16x spatial compression, specifically inspecting event edges and motion.

2. Multimodal RoPE:
   Event uses t/y/x, Action uses only shared t, camera is an embedding, and
   coordinate changes alter rotary phase.

3. Rectified flow:
   Event and Action share exactly one t; at t=0 X_t=X_0; at t=1 X_t=epsilon;
   target velocity is epsilon-X_0; all shapes are finite. An oracle returning
   epsilon-X_0 must recover X_0 at numerical precision through the complete
   inference scheduler, before output clamping (test multiple step counts).

4. Dataset alignment:
   clean Event input is V_k, first action is a_k, first Event target is
   V_{k+1}, and action a_k / Event V_{k+1} both have RoPE coordinate 1.

5. Before production experiments: real pretrained RGB/text GPU execution,
   real-event reconstruction against a zero baseline with region-specific
   metrics and visual inspection, then D512/L8/K8 GPU smoke/overfit. Missing CUDA
   must be reported as a skipped gate, never a production pass. Profile context
   preprocessing, DINO forward, whole-step time and peak memory before K=16/32.

## Checkpoint contract

New policy checkpoints use format=ewm_trainable_v1 and contain EWM weights and
buffers (including context projections and embeddings), excluding the frozen
DINO/T5 state. Frozen pretrained weights/tokenizers are explicit external
dependencies; retain their exact local snapshots. Save model config, step,
horizon and the normalization sidecar. Validate all EWM keys strictly when
loading. Legacy full-state checkpoints remain readable but are no longer the
save format. Policy checkpoints do not imply exact optimizer/RNG resume.

## Non-goals for v1

- Event VAE, KL loss, pretrained event latents, or latent-space flow.
- Independent Event and Action flow times.
- Clean/noisy role or segment embeddings.
- Causal Event decoder/cache.
- Structured/factorized Event attention before K=32 profiling.
- Adding DINO/T5 tokens to Event-Action mixed self-attention.


### Voxel-bin parameter

`voxel_bins` is external to the model and loader. v1 defaults to `4`; `1` is supported as an experiment with the same causal window and a codec branch whose temporal strides are all one.
