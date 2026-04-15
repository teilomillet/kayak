#!/usr/bin/env bash
set -euo pipefail

mode="${1:-aligned}"
state_file="${FIXTURE_STATE_FILE:?FIXTURE_STATE_FILE must be set}"
run_index=0

if [[ -f "${state_file}" ]]; then
  run_index="$(cat "${state_file}")"
fi
run_index="$((run_index + 1))"
printf '%s\n' "${run_index}" > "${state_file}"

section_one_label="alpha direct"
section_two_label="beta batch worker_count=4"
section_one_mean="1.0"
section_two_mean="4.0"

case "${mode}" in
  aligned)
    if [[ "${run_index}" -eq 2 ]]; then
      section_one_mean="3.0"
      section_two_mean="2.0"
    fi
    ;;
  mismatched)
    if [[ "${run_index}" -eq 2 ]]; then
      section_one_label="beta batch worker_count=4"
      section_two_label="alpha direct"
      section_one_mean="2.0"
      section_two_mean="3.0"
    fi
    ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    exit 1
    ;;
esac

printf '== %s ==\n' "${section_one_label}"
printf '%s\n' '--------------------------------------------------------------------------------'
printf '%s\n' 'Benchmark Report (s)'
printf '%s\n' '--------------------------------------------------------------------------------'
printf 'Mean: %s\n' "${section_one_mean}"
printf '%s\n' 'Total: 0.01'
printf '%s\n' 'Iters: 1'
printf '%s\n' 'Warmup Total: 0.0'
printf '%s\n' 'Fastest Mean: 0.0'
printf '%s\n' 'Slowest Mean: 0.0'
printf '\n'
printf '== %s ==\n' "${section_two_label}"
printf '%s\n' '--------------------------------------------------------------------------------'
printf '%s\n' 'Benchmark Report (s)'
printf '%s\n' '--------------------------------------------------------------------------------'
printf 'Mean: %s\n' "${section_two_mean}"
printf '%s\n' 'Total: 0.01'
printf '%s\n' 'Iters: 1'
printf '%s\n' 'Warmup Total: 0.0'
printf '%s\n' 'Fastest Mean: 0.0'
printf '%s\n' 'Slowest Mean: 0.0'
