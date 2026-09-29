#!/usr/bin/env bash
# Source from an Aurora login or compute node. All mutable state stays on Flare.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script: source ${BASH_SOURCE[0]}" >&2
    exit 2
fi

set +u
module load frameworks/2025.3.1
set -u

rfd_project_root="${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}"
rfd_env=${RFDIFFUSION_IDP_AURORA_ENV:-${rfd_project_root}/envs/rfdiffusion-idp-aurora}
rfd_cache=${RFDIFFUSION_IDP_CACHE_ROOT:-${rfd_project_root}/cache/rfdiffusion-idp-aurora}
rfd_runtime_home=${RFDIFFUSION_IDP_RUNTIME_HOME:-${rfd_cache}/runtime-home}

if [[ ! -x "${rfd_env}/bin/python" ]]; then
    echo "RFdiffusion IDP environment is missing: ${rfd_env}" >&2
    return 2
fi

mkdir -p \
    "${rfd_runtime_home}" \
    "${rfd_cache}/pip" \
    "${rfd_cache}/tmp" \
    "${rfd_cache}/xdg" \
    "${rfd_cache}/torch" \
    "${rfd_cache}/torchinductor" \
    "${rfd_cache}/torch_extensions" \
    "${rfd_cache}/matplotlib" \
    "${rfd_cache}/python-userbase"

export RFDIFFUSION_IDP_AURORA_ENV="${rfd_env}"
export RFDIFFUSION_IDP_CACHE_ROOT="${rfd_cache}"
export RFDIFFUSION_IDP_RUNTIME_HOME="${rfd_runtime_home}"
export HOME="${rfd_runtime_home}"
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${rfd_cache}/python-userbase"
export PIP_CACHE_DIR="${rfd_cache}/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TMPDIR="${rfd_cache}/tmp"
export XDG_CACHE_HOME="${rfd_cache}/xdg"
export TORCH_HOME="${rfd_cache}/torch"
export TORCHINDUCTOR_CACHE_DIR="${rfd_cache}/torchinductor"
export TORCH_EXTENSIONS_DIR="${rfd_cache}/torch_extensions"
export MPLCONFIGDIR="${rfd_cache}/matplotlib"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

source "${rfd_env}/bin/activate"
unset rfd_project_root rfd_env rfd_cache rfd_runtime_home
