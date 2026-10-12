#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
limit_bytes="${MAX_GIT_FILE_BYTES:-26214400}"
failed=0

while IFS= read -r -d '' relative_path; do
    path="${repo_root}/${relative_path}"
    [[ -f "${path}" ]] || continue
    size="$(stat -c '%s' "${path}")"
    if (( size > limit_bytes )); then
        printf 'oversized: %s bytes  %s\n' "${size}" "${relative_path}" >&2
        failed=1
    fi
done < <(git -C "${repo_root}" ls-files -z --cached --others --exclude-standard)

if (( failed != 0 )); then
    printf 'Refusing files larger than %s bytes. Use a release/object store plus SHA256 instead.\n' "${limit_bytes}" >&2
    exit 1
fi

printf 'Large-file check passed (limit: %s bytes).\n' "${limit_bytes}"
