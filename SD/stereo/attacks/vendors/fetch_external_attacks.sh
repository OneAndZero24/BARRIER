#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="${ROOT_DIR}/.tmp_fetch"
mkdir -p "${TMP_DIR}"

fetch_repo() {
  local owner_repo="$1"
  local out_dir="$2"
  local branch="$3"
  local url="https://codeload.github.com/${owner_repo}/tar.gz/refs/heads/${branch}"

  echo "Fetching ${owner_repo} (${branch})"
  if ! curl -L --fail "${url}" -o "${TMP_DIR}/${out_dir}.tar.gz"; then
    return 1
  fi

  rm -rf "${ROOT_DIR}/${out_dir}"
  mkdir -p "${ROOT_DIR}/${out_dir}"
  tar -xzf "${TMP_DIR}/${out_dir}.tar.gz" -C "${TMP_DIR}"

  local extracted
  extracted=$(find "${TMP_DIR}" -maxdepth 1 -type d -name "*$(basename "${owner_repo}")*" | head -n 1)
  if [[ -z "${extracted}" ]]; then
    echo "Could not locate extracted folder for ${owner_repo}" >&2
    return 1
  fi

  cp -R "${extracted}/." "${ROOT_DIR}/${out_dir}/"
  rm -rf "${extracted}"
}

# Try main first, then master.
fetch_with_fallback() {
  local owner_repo="$1"
  local out_dir="$2"
  if fetch_repo "${owner_repo}" "${out_dir}" "main"; then
    return 0
  fi
  fetch_repo "${owner_repo}" "${out_dir}" "master"
}

fetch_with_fallback "OPTML-Group/Diffusion-MU-Attack" "unlearndiffatk"
fetch_with_fallback "chiayi-hsu/Ring-A-Bell" "ring-a-bell"
fetch_with_fallback "NYU-DICE-Lab/circumventing-concept-erasure" "cce"

cat > "${ROOT_DIR}/VENDORED_ATTACKS.md" <<EOF
# External Attack Vendors

This directory vendors external attack code referenced by the STEREO paper for Table 2 style robustness evaluation.

Vendored repos:
- OPTML-Group/Diffusion-MU-Attack -> unlearndiffatk
- chiayi-hsu/Ring-A-Bell -> ring-a-bell
- NYU-DICE-Lab/circumventing-concept-erasure -> cce

Refresh command:
  bash SD/stereo/attacks/vendors/fetch_external_attacks.sh
EOF

rm -rf "${TMP_DIR}"

echo "Done. Vendored attacks are available under: ${ROOT_DIR}"
