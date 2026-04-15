#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_file_contains() {
  local path="$1"
  local pattern="$2"
  if ! rg -n --fixed-strings "${pattern}" "${path}" > /dev/null; then
    fail "expected ${path} to contain: ${pattern}"
  fi
}

run_case() {
  local mode="$1"
  local tmp_dir
  local log_root
  local state_file
  local output
  local run_dir

  tmp_dir="$(mktemp -d "/tmp/kayak-run-bench-quiet-${mode}.XXXXXX")"
  log_root="${tmp_dir}/bench_quiet"
  state_file="${tmp_dir}/fixture_state.txt"

  output="$(
    cd "${repo_root}"
    KAYAK_BENCH_QUIET_LOG_ROOT="${log_root}" \
      FIXTURE_STATE_FILE="${state_file}" \
      bash scripts/run_bench_quiet.sh \
        --repeats 2 \
        --max-other-cpu 100000 \
        --quiet-checks 1 \
        --sleep-seconds 0 \
        --timeout-seconds 1 \
        -- \
        bash tests/fixtures/run_bench_quiet_multisection_fixture.sh "${mode}"
  )"
  run_dir="$(printf '%s\n' "${output}" | awk -F': ' '/^quiet bench log dir:/ { print $2; exit }')"
  [[ -n "${run_dir}" ]] || fail "could not parse run directory for mode=${mode}"
  [[ -d "${run_dir}" ]] || fail "missing run directory ${run_dir}"

  case "${mode}" in
    aligned)
      [[ -f "${run_dir}/section_summary.tsv" ]] || fail "missing section summary TSV"
      [[ -f "${run_dir}/section_summary.txt" ]] || fail "missing section summary text"
      assert_file_contains "${run_dir}/summary.txt" "multi_section=1"
      assert_file_contains "${run_dir}/summary.txt" "section_count=2"
      assert_file_contains "${run_dir}/section_summary.tsv" $'1\talpha direct\t1.0\t3.0\t2\t1.0\t2\t2\t3.0'
      assert_file_contains "${run_dir}/section_summary.tsv" $'2\tbeta batch worker_count=4\t4.0\t2.0\t2\t2.0\t3\t3\t4.0'
      ;;
    mismatched)
      [[ ! -f "${run_dir}/section_summary.tsv" ]] || fail "unexpected section summary TSV for mismatched sections"
      assert_file_contains "${run_dir}/summary.txt" "section_alignment=failed"
      [[ -s "${run_dir}/section_alignment_error.txt" ]] || fail "missing alignment error details"
      ;;
    *)
      fail "unsupported mode=${mode}"
      ;;
  esac

  rm -rf "${tmp_dir}"
}

run_case aligned
run_case mismatched
