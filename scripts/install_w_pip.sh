#!/usr/bin/env bash
set -euo pipefail

with_astrasim=1

for arg in "$@"; do
    case "$arg" in
        --with-astrasim)
            with_astrasim=1
            ;;
        -h|--help)
            echo "Usage: $0 [--with-astrasim]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON:-python3}"

"${python_bin}" -m pip install -e "${repo_root}"

if [[ "${with_astrasim}" -eq 1 ]]; then
    "${repo_root}/scripts/install_astrasim.sh"
fi
