#!/usr/bin/env bash
# Recreate the Aurora RFdiffusion IDP environment without CUDA packages.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}"
repo_root=${RFDIFFUSION_IDP_REPO:-$(cd -- "${script_dir}/.." && pwd)}
env_root=${RFDIFFUSION_IDP_AURORA_ENV:-${project_root}/envs/rfdiffusion-idp-aurora}
cache_root=${RFDIFFUSION_IDP_CACHE_ROOT:-${project_root}/cache/rfdiffusion-idp-aurora}
weights=0

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf 'Usage: aurora/install_aurora.sh [--weights]\n'
    printf 'Install RFdiffusion IDP on Aurora; --weights downloads and verifies its checkpoints.\n'
    exit 0
elif [[ ${1:-} == "--weights" ]]; then
    weights=1
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--weights]" >&2
    exit 2
fi

frameworks_module="${AURORA_FRAMEWORKS_MODULE:-frameworks}"
set +u
module load "${frameworks_module}"
set -u
mkdir -p "${cache_root}/runtime-home" "${cache_root}/pip" "${cache_root}/tmp" "${cache_root}/xdg" "${cache_root}/python-userbase"
export HOME="${cache_root}/runtime-home"
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${cache_root}/python-userbase"
export PIP_CACHE_DIR="${cache_root}/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="${cache_root}/tmp"
export XDG_CACHE_HOME="${cache_root}/xdg"

test -x "${env_root}/bin/python" || python -m venv --system-site-packages "${env_root}"
source "${env_root}/bin/activate"
python -m pip install --no-cache-dir e3nn==0.3.3 opt-einsum pyrsistent
python -m pip install --no-cache-dir -r "${script_dir}/requirements-aurora.txt"
python -m pip install --no-deps -e "${repo_root}/env/SE3Transformer"
python -m pip install --no-deps -e "${repo_root}"

if [[ ${weights} -eq 1 ]]; then
    model_root="${cache_root}/models"
    mkdir -p "${model_root}"
    curl -fL --retry 3 --continue-at - -o "${model_root}/InpaintSeq_ckpt.pt" \
        https://zenodo.org/api/records/15453428/files/InpaintSeq_ckpt.pt/content
    curl -fL --retry 3 --continue-at - -o "${model_root}/InpaintSeq_Fold_ckpt.pt" \
        https://zenodo.org/api/records/15453428/files/InpaintSeq_Fold_ckpt.pt/content
    printf '%s  %s\n' \
        a6f8652938bb45c332ffa683d8ad3509 "${model_root}/InpaintSeq_ckpt.pt" \
        1e9245a486262dff3cb3286f22a3014d "${model_root}/InpaintSeq_Fold_ckpt.pt" | md5sum -c -
    test -e "${repo_root}/models" || ln -s "${model_root}" "${repo_root}/models"
fi

echo "RFdiffusion IDP Aurora environment ready: ${env_root}"
