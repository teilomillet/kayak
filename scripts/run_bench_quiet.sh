#!/usr/bin/env bash
set -euo pipefail

repeats=3
max_other_cpu=40
quiet_checks=3
sleep_seconds=2
timeout_seconds=120
force=0
log_root=".cache/kayak/bench_quiet"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_bench_quiet.sh [options] -- <command...>

Options:
  --repeats N            Number of benchmark runs to execute. Default: 3
  --max-other-cpu N      Maximum allowed aggregate %CPU from other processes
                         before each run. Default: 40
  --quiet-checks N       Consecutive quiet samples required. Default: 3
  --sleep-seconds N      Delay between quiet checks. Default: 2
  --timeout-seconds N    Maximum wait for a quiet machine. Default: 120
  --force                Run even if the quiet wait times out
  -h, --help             Show this help

Notes:
  - This script does not reserve CPU cores. It improves comparison quality by
    waiting for a quieter host, snapshotting competing processes, and reporting
    a median across repeated runs.
  - The aggregate "%CPU" is sampled from `ps` and represents observed host
    activity from processes other than the benchmark wrapper itself.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeats)
      repeats="$2"
      shift 2
      ;;
    --max-other-cpu)
      max_other_cpu="$2"
      shift 2
      ;;
    --quiet-checks)
      quiet_checks="$2"
      shift 2
      ;;
    --sleep-seconds)
      sleep_seconds="$2"
      shift 2
      ;;
    --timeout-seconds)
      timeout_seconds="$2"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing benchmark command." >&2
  usage >&2
  exit 1
fi

cmd=("$@")
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${log_root}/${timestamp}"
mkdir -p "${run_dir}"

snapshot_top() {
  ps -ax -o pid=,pcpu=,pmem=,command= | sort -k2 -nr | awk 'NR <= 12'
}

other_cpu_load() {
  ps -ax -o pid=,pcpu=,command= | awk -v self="$$" '
    $1 != self && $2 + 0 >= 2 && $3 != "ps" && $3 != "awk" && $3 != "sort" && $3 != "head" {
      sum += $2
    }
    END {
      printf "%.2f\n", sum + 0
    }
  '
}

wait_for_quiet() {
  local run_index="$1"
  local stable=0
  local sample=0
  local start_epoch
  local now_epoch
  local load
  local snapshot_file

  start_epoch="$(date +%s)"

  while true; do
    load="$(other_cpu_load)"
    snapshot_file="${run_dir}/quiet_check_run_${run_index}_sample_${sample}.txt"
    snapshot_top > "${snapshot_file}"

    printf \
      'quiet-check run=%s sample=%s other_cpu=%s threshold=%s\n' \
      "${run_index}" "${sample}" "${load}" "${max_other_cpu}"

    if awk -v load="${load}" -v threshold="${max_other_cpu}" 'BEGIN { exit !(load <= threshold) }'; then
      stable=$((stable + 1))
      if [[ "${stable}" -ge "${quiet_checks}" ]]; then
        return 0
      fi
    else
      stable=0
    fi

    now_epoch="$(date +%s)"
    if (( now_epoch - start_epoch >= timeout_seconds )); then
      echo "quiet wait timed out after ${timeout_seconds}s" >&2
      echo "latest competing-process snapshot:" >&2
      cat "${snapshot_file}" >&2
      if [[ "${force}" -eq 1 ]]; then
        return 0
      fi
      return 1
    fi

    sample=$((sample + 1))
    sleep "${sleep_seconds}"
  done
}

summarize_means() {
  printf '%s\n' "$@" | sort -g | awk '
    {
      values[++n] = $1
      sum += $1
    }
    END {
      if (n == 0) {
        exit 1
      }
      min = values[1]
      max = values[n]
      mean = sum / n
      if (n % 2 == 1) {
        median = values[(n + 1) / 2]
      } else {
        median = (values[n / 2] + values[(n / 2) + 1]) / 2
      }
      printf "summary runs=%d min=%s median=%s mean=%s max=%s\n", n, min, median, mean, max
    }
  '
}

{
  echo "timestamp_utc=${timestamp}"
  echo "command=${cmd[*]}"
  echo "repeats=${repeats}"
  echo "max_other_cpu=${max_other_cpu}"
  echo "quiet_checks=${quiet_checks}"
  echo "sleep_seconds=${sleep_seconds}"
  echo "timeout_seconds=${timeout_seconds}"
} > "${run_dir}/meta.txt"

echo "quiet bench log dir: ${run_dir}"
echo "benchmark command: ${cmd[*]}"

means=()
single_metric_runs=1

for run_index in $(seq 1 "${repeats}"); do
  wait_for_quiet "${run_index}"

  before_file="${run_dir}/run_${run_index}_before.txt"
  output_file="${run_dir}/run_${run_index}.txt"
  after_file="${run_dir}/run_${run_index}_after.txt"

  snapshot_top > "${before_file}"
  echo "run ${run_index}/${repeats}"
  "${cmd[@]}" | tee "${output_file}"
  snapshot_top > "${after_file}"

  run_means=()
  while IFS= read -r parsed_mean; do
    run_means+=("${parsed_mean}")
  done < <(awk '/^Mean: / { print $2 }' "${output_file}")
  if [[ "${#run_means[@]}" -eq 0 ]]; then
    echo "Could not parse benchmark Mean: from ${output_file}" >&2
    exit 1
  fi

  printf '%s\n' "${run_means[@]}" > "${run_dir}/run_${run_index}_means.txt"

  if [[ "${#run_means[@]}" -eq 1 ]]; then
    means+=("${run_means[0]}")
    echo "parsed_mean=${run_means[0]}"
  else
    single_metric_runs=0
    echo "parsed_mean_count=${#run_means[@]}"
  fi
done

if [[ "${single_metric_runs}" -eq 1 ]]; then
  printf '%s\n' "${means[@]}" > "${run_dir}/means.txt"
  summary="$(summarize_means "${means[@]}")"
  echo "${summary}"
  echo "${summary}" > "${run_dir}/summary.txt"
else
  summary="summary runs=${repeats} multi_section=1 scalar_summary=skipped log_dir=${run_dir}"
  echo "${summary}"
  echo "${summary}" > "${run_dir}/summary.txt"
fi
