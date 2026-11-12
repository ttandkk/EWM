"""Versioned project assets and explicitly external simulator dependencies."""
__all__ = ["asset_path", "upstream_asset_path"]


def __getattr__(name):
    if name in __all__:
        from . import resolver
        return getattr(resolver, name)
    raise AttributeError(name)
