#!/bin/bash

# Benchmark script for measuring command execution time (in milliseconds)
# Usage: ./benchmark.sh [options] -- <command> [args...]

set -e

WARMUP_RUNS=3
BENCHMARK_RUNS=10
QUIET=false

# Parse options
while [[ $# -gt 0 ]]; do
  case $1 in
    --warmup=*) WARMUP_RUNS="${1#*=}"; shift ;;
    --runs=*) BENCHMARK_RUNS="${1#*=}"; shift ;;
    -q|--quiet) QUIET=true; shift ;;
    -h|--help)
      echo "Usage: $0 [options] -- <command> [args...]"
      echo "Options: --warmup=N, --runs=N, -q, -h"
      exit 0
      ;;
    --) shift; break ;;
    *) break ;;
  esac
done

[[ $# -eq 0 ]] && { echo "Error: No command provided." >&2; exit 1; }
COMMAND=("$@")

# Convert time string to milliseconds
to_ms() {
  local t="$1"
  local seconds=0
  
  # Handle various formats and convert to seconds
  if [[ "$t" =~ ^[0-9]+:[0-9]+\.[0-9]+$ ]]; then
    # mm:ss.ff format
    local m=$(echo "$t" | cut -d: -f1)
    local s=$(echo "$t" | cut -d: -f2)
    seconds=$(echo "$m * 60 + $s" | bc)
  elif [[ "$t" =~ ^[0-9]+m[0-9]+\.[0-9]+s?$ ]]; then
    # 0m0.106s format
    local m=$(echo "$t" | sed 's/m.*//')
    local s=$(echo "$t" | sed 's/.*m//; s/s$//')
    seconds=$(echo "$m * 60 + $s" | bc)
  else
    # Assume already in seconds (numeric)
    seconds="$t"
  fi
  
  # Convert to milliseconds
  echo "$seconds * 1000" | bc
}

# Measure execution time
run_with_time() {
  local redirect=""
  [[ "$QUIET" == true ]] && redirect=">/dev/null 2>&1"
  
  if command -v /usr/bin/time >/dev/null 2>&1; then
    local elapsed=$(/usr/bin/time -f "%e" -- "${COMMAND[@]}" $redirect 2>&1)
    to_ms "$elapsed"
  else
    local elapsed=$( { time "${COMMAND[@]}" $redirect 2>&1; } 2>&1 | grep real | awk '{print $2}')
    to_ms "$elapsed"
  fi
}

# Print header
[[ "$QUIET" == false ]] && {
  echo "=========================================="
  echo "Benchmark: ${COMMAND[*]}"
  echo "Warmup: $WARMUP_RUNS | Runs: $BENCHMARK_RUNS"
  echo "=========================================="
  echo "Warming up..."
}

# Warmup
for ((i=1; i<=WARMUP_RUNS; i++)); do
  [[ "$QUIET" == false ]] && echo -n "  [$i/$WARMUP_RUNS] "
  "${COMMAND[@]}" >/dev/null 2>&1 || true
  [[ "$QUIET" == false ]] && echo "done"
done

[[ "$QUIET" == false ]] && echo -e "\nRunning benchmarks..."

# Benchmark runs
declare -a times
for ((i=1; i<=BENCHMARK_RUNS; i++)); do
  [[ "$QUIET" == false ]] && echo -n "  [$i/$BENCHMARK_RUNS] "
  elapsed_ms=$(run_with_time)
  times+=("$elapsed_ms")
  [[ "$QUIET" == false ]] && printf "%.2f ms\n" "$elapsed_ms"
done

# Calculate statistics
IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
unset IFS

sum=0
for t in "${times[@]}"; do
  sum=$(echo "$sum + $t" | bc)
done

mean=$(echo "scale=2; $sum / $BENCHMARK_RUNS" | bc)
median="${sorted[$((BENCHMARK_RUNS/2))]}"
min="${sorted[0]}"
max="${sorted[$((BENCHMARK_RUNS-1))]}"

# Standard deviation
sum_sq=0
for t in "${times[@]}"; do
  diff=$(echo "$t - $mean" | bc)
  sum_sq=$(echo "$sum_sq + $diff * $diff" | bc)
done
stddev=$(echo "scale=2; sqrt($sum_sq / $BENCHMARK_RUNS)" | bc)

# Print results
[[ "$QUIET" == false ]] && echo ""
echo "=========================================="
echo "Results (ms):"
echo "=========================================="
printf "  Mean:   %8.2f ms\n" "$mean"
printf "  Median: %8.2f ms\n" "$median"
printf "  Min:    %8.2f ms\n" "$min"
printf "  Max:    %8.2f ms\n" "$max"
printf "  StdDev: %8.2f ms\n" "$stddev"
echo "=========================================="
