"""Resolve the configured video encoder without depending on a model environment."""
import os
import shutil


def find_ffmpeg():
    explicit = os.environ.get("PIPER_FFMPEG")
    if explicit:
        found = shutil.which(explicit)
        if found is None:
            raise FileNotFoundError(f"PIPER_FFMPEG is not executable: {explicit}")
        return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Install ffmpeg or set PIPER_FFMPEG to its executable") from exc
