#!/usr/bin/env bash

trim_env_value() {
  local value="${1-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

strip_env_quotes() {
  local value
  value="$(trim_env_value "${1-}")"
  if [[ ${#value} -ge 2 ]]; then
    local first="${value:0:1}"
    local last="${value: -1}"
    if [[ "${first}" == "${last}" && ( "${first}" == "'" || "${first}" == '"' ) ]]; then
      value="${value:1:${#value}-2}"
      value="$(trim_env_value "${value}")"
    fi
  fi
  printf '%s' "${value}"
}

load_env_file_if_present() {
  local env_path="${1-}"
  if [[ -z "${env_path}" || ! -f "${env_path}" ]]; then
    return 0
  fi

  local raw_line line key value
  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%$'\r'}"
    line="$(trim_env_value "${line}")"
    if [[ -z "${line}" || "${line:0:1}" == "#" ]]; then
      continue
    fi
    if [[ "${line}" == export\ * ]]; then
      line="$(trim_env_value "${line#export }")"
    fi
    if [[ "${line}" != *=* ]]; then
      continue
    fi

    key="$(trim_env_value "${line%%=*}")"
    value="$(trim_env_value "${line#*=}")"
    if [[ -z "${key}" || ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi
    if [[ -n "${!key+x}" ]]; then
      continue
    fi

    value="$(strip_env_quotes "${value}")"
    printf -v "${key}" '%s' "${value}"
    export "${key}"
  done < "${env_path}"
}

require_env() {
  local key="${1}"
  if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[ERROR] Invalid environment variable name: ${key}" >&2
    return 1
  fi
  if [[ -z "${!key-}" ]]; then
    echo "[ERROR] Required environment variable is not set: ${key}" >&2
    return 1
  fi
}

require_envs() {
  local key
  for key in "$@"; do
    require_env "${key}" || return 1
  done
}

resolve_repo_path() {
  local repo_root="${1}"
  local raw_path="${2-}"
  if [[ -z "${raw_path}" ]]; then
    printf '\n'
    return 0
  fi
  if [[ "${raw_path}" == /* ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi
  printf '%s\n' "${repo_root%/}/${raw_path#./}"
}

resolve_env_path_var() {
  local repo_root="${1}"
  local key="${2}"
  require_env "${key}" || return 1
  local resolved
  resolved="$(resolve_repo_path "${repo_root}" "${!key}")"
  printf -v "${key}" '%s' "${resolved}"
  export "${key}"
}

resolve_env_path_if_set_var() {
  local repo_root="${1}"
  local key="${2}"
  if [[ -z "${!key-}" ]]; then
    return 0
  fi
  local resolved
  resolved="$(resolve_repo_path "${repo_root}" "${!key}")"
  printf -v "${key}" '%s' "${resolved}"
  export "${key}"
}

is_truthy() {
  local value="${1-}"
  value="$(trim_env_value "${value}")"
  value="${value,,}"
  [[ -n "${value}" && "${value}" != "0" && "${value}" != "false" && "${value}" != "no" && "${value}" != "off" ]]
}

require_file() {
  local path="${1}"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file not found: ${path}" >&2
    exit 1
  fi
}

require_nonempty_file() {
  local path="${1}"
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] Required non-empty file not found (or empty): ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="${1}"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 1
  fi
}
