"""Explicit, in-memory relocation of model resources, never checkpoint rewriting."""
from __future__ import annotations

import copy
import os
from pathlib import Path

# Only these documented fields are paths. Model IDs and arbitrary strings are
# intentionally excluded: replacing all strings silently changes model behavior.
EWM_RESOURCE_FIELDS = ("dino_path", "text_path", "event_vae_path")
OPENWAM_RESOURCE_FIELDS = (
    "model.video_backbone.model_path", "model.video_backbone.vae_path",
    "model.video_backbone.text_encoder_path", "model.video_backbone.tokenizer_path",
    "model.action_backbone.pretrained_path", "model.vlm_backbone.checkpoint_path",
)
FASTWAM_RESOURCE_FIELDS = ("model.action_dit_pretrained_path",)


class ResourceResolver:
    def __init__(self, *, project_root=None, pretrained_root=None, path_maps=()):
        self.project_root = Path(project_root).expanduser().resolve() if project_root else None
        self.pretrained_root = Path(pretrained_root).expanduser().resolve() if pretrained_root else None
        self.maps = []
        for entry in path_maps:
            old, sep, new = entry.partition("=")
            if not sep or not old or not new:
                raise ValueError("path maps must be OLD_ROOT=NEW_ROOT")
            old_path = Path(old).expanduser()
            if not old_path.is_absolute():
                raise ValueError("path map source must be absolute")
            self.maps.append((old_path, Path(new).expanduser().resolve()))
        self.maps.sort(key=lambda pair: len(pair[0].parts), reverse=True)

    def resolve(self, value, *, base=None):
        if value is None or value == "":
            return value
        path = Path(os.path.expandvars(str(value))).expanduser()
        if path.is_absolute():
            for old, new in self.maps:
                try:
                    return str(new / path.relative_to(old))
                except ValueError:
                    pass
        # Explicit maps take priority; conventional pretrained suffixes may
        # also occur in relative saved paths such as ../pretrained/Wan-AI.
        if self.pretrained_root and "pretrained" in path.parts:
            index = path.parts.index("pretrained")
            return str(self.pretrained_root.joinpath(*path.parts[index + 1:]))
        if path.is_absolute():
            return str(path)
        anchor = Path(base) if base is not None else self.project_root
        return str((anchor / path).resolve()) if anchor else str(path)

    def ewm_model_config(self, config):
        result = copy.deepcopy(dict(config))
        for key in EWM_RESOURCE_FIELDS:
            if key in result:
                result[key] = self.resolve(result[key])
        return result

    def config(self, cfg, family):
        from omegaconf import OmegaConf
        result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        fields = OPENWAM_RESOURCE_FIELDS if family == "openwam" else FASTWAM_RESOURCE_FIELDS
        for key in fields:
            value = OmegaConf.select(result, key)
            if value is not None:
                OmegaConf.update(result, key, self.resolve(value))
        if family == "openwam":
            components = OmegaConf.select(result, "model.video_backbone.components", default=[])
            for component in components:
                for key in ("path", "model_path", "tokenizer_path", "config_path"):
                    if component.get(key) is not None:
                        component[key] = self.resolve(component[key])
        return result
