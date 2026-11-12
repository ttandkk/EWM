"""Read legacy policies only when their removed features were already disabled."""


def normalize_policy_payload(payload):
    """Drop provably unused legacy weights; keep strict core weight validation."""
    config = dict(payload['model_config'])
    mode = config.get('training_mode', 'joint')
    format_name = payload.get('format')
    accepted = {
        'joint': {'ewm_wan_latent_v2', 'ewm_rgb_event_memory_120hz_v2'},
        'event_only': {'ewm_wan_latent_v2', 'ewm_rgb_event_memory_120hz_v2'},
        'rgb_action': {'ewm_rgb_action_latent_v1'},
        'action_only': {'ewm_action_only_v1', 'ewm_action_only_v2'},
    }
    if format_name not in accepted.get(mode, set()):
        raise ValueError('incompatible EWM checkpoint format and training_mode combination')
    legacy_memory = format_name == 'ewm_rgb_event_memory_120hz_v2'
    if legacy_memory != (config.get('type', 'ewm') == 'ewm_rgb_event_memory'):
        raise ValueError('checkpoint format and model type must match')
    if config.get('type', 'ewm') not in ('ewm', 'ewm_rgb_event_memory'):
        raise ValueError('unsupported EWM model type')
    if (config.get('memory_enabled', legacy_memory) or config.get('rgb_history_steps', 0)
            or config.get('alignment_weight', 0) or (config.get('window_routing') or {}).get('enabled', False)):
        raise ValueError('checkpoint enables removed memory/history/alignment/routing features; use its archived source')
    for key in list(config):
        if key.startswith(('memory_', 'alignment_')) or key in ('rgb_history_steps', 'window_routing', 'activity_weight'):
            del config[key]
    config['type'] = 'ewm'
    state = dict(payload['model'])
    for key in list(state):
        unused_memory = legacy_memory and key.startswith(('rgb_event_memory.', 'memory_projectors.', 'memory_key_norms.'))
        unused_event = format_name == 'ewm_action_only_v1' and (
            key == 'camera' or key.startswith(('event_input.', 'event_head.'))
            or (key.startswith('blocks.') and '.event.' in key))
        if unused_memory or unused_event:
            del state[key]
    return {**payload, 'model_config': config, 'model': state}
