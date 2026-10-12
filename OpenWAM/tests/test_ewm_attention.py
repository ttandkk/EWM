"""Contract tests for EWM's full Event/Action mixed attention."""

import pytest
import torch


def test_full_mixed_attention_is_bidirectional_and_preserves_stream_shapes():
    """Both query streams attend the complete Event+Action sequence."""
    from openwam.model.architectures.ewm import full_mixed_self_attention

    q_event = torch.zeros(1, 2, 4)
    k_event = torch.zeros_like(q_event)
    v_event = torch.zeros_like(q_event)
    q_action = torch.zeros(1, 1, 4)
    k_action = torch.zeros_like(q_action)
    v_action = torch.full_like(q_action, 2.0)
    out_event, out_action = full_mixed_self_attention(
        q_event, k_event, v_event, q_action, k_action, v_action, num_heads=2
    )
    assert out_event.shape == q_event.shape
    assert out_action.shape == q_action.shape
    assert torch.allclose(out_event, torch.full_like(out_event, 2 / 3))
    assert torch.allclose(out_action, torch.full_like(out_action, 2 / 3))


def test_full_mixed_attention_requires_shared_head_geometry():
    from openwam.model.architectures.ewm import full_mixed_self_attention

    x = torch.zeros(1, 1, 6)
    with pytest.raises(ValueError, match="divisible"):
        full_mixed_self_attention(x, x, x, x, x, x, num_heads=4)


def test_reverse_visibility_and_gradients():
    from openwam.model.architectures.ewm import full_mixed_self_attention

    e = torch.randn(2, 3, 16, requires_grad=True)
    a = torch.randn(2, 2, 16, requires_grad=True)
    oe, oa = full_mixed_self_attention(e, e, e, a, a, a, num_heads=2)
    assert torch.autograd.grad(oa.sum(), e, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(oe.sum(), a)[0].abs().sum() > 0


def test_shared_rope_alignment_and_axis_phases():
    from openwam.model.architectures.ewm import event_action_coordinates, multimodal_rope

    ec, ac = event_action_coordinates(8, 16, 16)
    assert ec.shape == (4608, 3)
    assert ac.shape == (9, 3)
    assert torch.equal(ec[512], ac[1])
    assert ec[512, 0] == 1
    assert torch.equal(ec[:256], ec[256:512])  # cameras use embeddings
    x = torch.randn(1, 2, 9, 64)
    rotated = multimodal_rope(x, ac)
    torch.testing.assert_close(rotated[..., 16:], x[..., 16:])
    assert not torch.allclose(rotated[..., :16], x[..., :16])
    for axis, (start, end) in enumerate(((0, 16), (16, 40), (40, 64))):
        coords = torch.zeros(9, 3)
        coords[:, axis] = 1
        out = multimodal_rope(x, coords)
        assert not torch.allclose(out[..., start:end], x[..., start:end])
        torch.testing.assert_close(out.square().sum(-1), x.square().sum(-1))


def test_p_major_layout():
    from openwam.model.architectures.ewm import canonical_voxel

    x = torch.arange(8.0).view(1, 1, 1, 8, 1, 1).expand(2, 3, 2, 8, 16, 16)
    v = canonical_voxel(x)
    assert v.shape == (2, 3, 2, 2, 4, 16, 16)
    assert torch.equal(v[0, 0, 0, :, :, 0, 0], torch.arange(8.0).reshape(2, 4))


def test_single_sdpa_matches_two_query_reference_gradients(monkeypatch):
    import torch.nn.functional as F

    from openwam.model.architectures.ewm import event_action_coordinates, full_mixed_self_attention, multimodal_rope

    torch.manual_seed(81)
    inputs = [torch.randn(2, n, 16, dtype=torch.float64, requires_grad=True) for n in (4, 4, 4, 2, 2, 2)]
    ec, ac = event_action_coordinates(1, 1, 1)
    qe, ke, ve, qa, ka, va = [x.unflatten(-1, (2, 8)).transpose(1, 2) for x in inputs]
    qe, ke = [multimodal_rope(x, ec) for x in (qe, ke)]
    qa, ka = [multimodal_rope(x, ac) for x in (qa, ka)]
    key, value = torch.cat((ke, ka), 2), torch.cat((ve, va), 2)
    expected = [F.scaled_dot_product_attention(q, key, value).transpose(1, 2).flatten(2) for q in (qe, qa)]
    original = F.scaled_dot_product_attention
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", counted)
    actual = full_mixed_self_attention(*inputs, num_heads=2, event_coordinates=ec, action_coordinates=ac)
    assert len(calls) == 1
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(sum(x.square().sum() for x in actual), inputs, retain_graph=True)
    gb = torch.autograd.grad(sum(x.square().sum() for x in expected), inputs)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
