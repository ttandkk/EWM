"""Strict classification of the deliberately pinned runtime metadata conflicts."""
from .config import load_lock


def classify_pip_check(output, returncode, profile, lock=None):
    known = {entry["pip_check_line"]: entry for entry in (lock or load_lock())["known_metadata_exceptions"]
             if entry["profile"] == profile}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if returncode == 0 and lines == ["No broken requirements found."]:
        return {"passed": True, "known_exceptions": [], "unexpected": [], "output": output, "returncode": returncode}
    matched = [known[line] for line in lines if line in known]
    unexpected = [line for line in lines if line not in known]
    return {"passed": bool(lines) and not unexpected and returncode == 1,
            "known_exceptions": matched, "unexpected": unexpected, "output": output, "returncode": returncode}
