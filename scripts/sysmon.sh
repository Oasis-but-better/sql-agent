#!/usr/bin/env bash
# CP5: system-resource monitor + throttle watchdog
# Runs every 10s, writes docs/sysstats.json, applies throttle after 3 bad ticks.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STATS_FILE="$PROJECT_ROOT/docs/sysstats.json"
LOG_FILE="$SCRIPT_DIR/sysmon.log"

bad_ticks=0
throttle_state="none"
throttle_note=""
empty_ticks=0

log() {
    echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG_FILE"
}

log "sysmon started pid=$$"

while true; do
    ts=$(date -u +%H:%M:%SZ)

    # --- RAM ---
    total_ram_gb=$(( $(sysctl -n hw.memsize) / 1000000000 ))

    # --- Memory free % ---
    mem_free_pct=$(memory_pressure 2>/dev/null | grep -i "System-wide memory free percentage" | grep -oE '[0-9]+' | head -1)
    if [[ -z "$mem_free_pct" ]]; then
        mem_free_pct=0
    fi

    if (( mem_free_pct < 10 )); then
        mem_pressure="critical"
    elif (( mem_free_pct < 25 )); then
        mem_pressure="warn"
    else
        mem_pressure="normal"
    fi

    mem_used_gb=$(awk "BEGIN {printf \"%.2f\", $total_ram_gb * (1 - $mem_free_pct/100)}")

    # --- Swap used GB ---
    swap_raw=$(sysctl -n vm.swapusage 2>/dev/null || echo "")
    swap_used_gb="null"
    if [[ -n "$swap_raw" ]]; then
        # format: "total = 2048.00M  used = 512.50M  free = 1535.50M"
        swap_str=$(echo "$swap_raw" | grep -oE 'used = [0-9]+\.[0-9]+[MG]' | grep -oE '[0-9]+\.[0-9]+[MG]')
        if [[ -n "$swap_str" ]]; then
            swap_num=$(echo "$swap_str" | grep -oE '[0-9]+\.[0-9]+')
            swap_unit=$(echo "$swap_str" | grep -oE '[MG]$')
            if [[ "$swap_unit" == "M" ]]; then
                swap_used_gb=$(awk "BEGIN {printf \"%.3f\", $swap_num / 1024}")
            else
                swap_used_gb=$(awk "BEGIN {printf \"%.3f\", $swap_num}")
            fi
        fi
    fi

    # --- Thermal ---
    cpu_speed_limit_pct="null"
    pmset_out=$(pmset -g therm 2>/dev/null || echo "")
    if [[ -n "$pmset_out" ]]; then
        limit=$(echo "$pmset_out" | grep -i "CPU_Speed_Limit" | grep -oE '[0-9]+' | head -1)
        if [[ -n "$limit" ]]; then
            cpu_speed_limit_pct="$limit"
        fi
    fi

    # --- Load ---
    load_avg=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    [[ -z "$load_avg" ]] && load_avg="null"

    # --- Training process ---
    pid=$(pgrep -f mlx_lm.lora 2>/dev/null | head -1 || true)
    train_cpu_pct="null"
    train_rss_gb="null"
    pid_json="null"

    if [[ -n "$pid" ]]; then
        pid_json="$pid"
        ps_out=$(ps -o %cpu=,rss= -p "$pid" 2>/dev/null || echo "")
        if [[ -n "$ps_out" ]]; then
            train_cpu_pct=$(echo "$ps_out" | awk '{printf "%.1f", $1}')
            rss_kb=$(echo "$ps_out" | awk '{print $2}')
            train_rss_gb=$(awk "BEGIN {printf \"%.3f\", $rss_kb / 1000000}")
        fi
        empty_ticks=0
    else
        empty_ticks=$(( empty_ticks + 1 ))
    fi

    # --- Watchdog ---
    is_bad=0
    if [[ "$mem_pressure" == "critical" ]]; then
        is_bad=1
    fi
    if [[ "$cpu_speed_limit_pct" != "null" ]] && (( cpu_speed_limit_pct < 70 )); then
        is_bad=1
    fi

    if (( is_bad )); then
        bad_ticks=$(( bad_ticks + 1 ))
    else
        bad_ticks=0
    fi

    if (( bad_ticks >= 3 )) && [[ -n "$pid" ]] && [[ "$throttle_state" == "none" ]]; then
        taskpolicy -b -p "$pid" 2>/dev/null && log "taskpolicy -b applied to pid=$pid" || log "taskpolicy failed for pid=$pid"
        renice +10 -p "$pid" 2>/dev/null && log "renice +10 applied to pid=$pid" || log "renice failed for pid=$pid"
        throttle_state="background-qos+renice"
        throttle_note="applied @${ts}: sustained pressure/thermal"
        log "throttle applied: $throttle_note"
    fi

    # --- Serialize throttle_note for JSON ---
    if [[ -z "$throttle_note" ]]; then
        throttle_note_json="null"
    else
        throttle_note_json="\"${throttle_note}\""
    fi

    # --- Write JSON ---
    cat > "$STATS_FILE" <<EOF
{
  "ts": "$ts",
  "total_ram_gb": $total_ram_gb,
  "mem_free_pct": $mem_free_pct,
  "mem_pressure": "$mem_pressure",
  "mem_used_gb": $mem_used_gb,
  "swap_used_gb": $swap_used_gb,
  "cpu_speed_limit_pct": $cpu_speed_limit_pct,
  "load_avg_1m": $load_avg,
  "train_pid": $pid_json,
  "train_cpu_pct": $train_cpu_pct,
  "train_rss_gb": $train_rss_gb,
  "throttle_state": "$throttle_state",
  "throttle_note": $throttle_note_json
}
EOF

    # --- Exit condition: training done for 2 ticks ---
    if (( empty_ticks >= 2 )); then
        throttle_note="training ended"
        cat > "$STATS_FILE" <<EOF
{
  "ts": "$ts",
  "total_ram_gb": $total_ram_gb,
  "mem_free_pct": $mem_free_pct,
  "mem_pressure": "$mem_pressure",
  "mem_used_gb": $mem_used_gb,
  "swap_used_gb": $swap_used_gb,
  "cpu_speed_limit_pct": $cpu_speed_limit_pct,
  "load_avg_1m": $load_avg,
  "train_pid": null,
  "train_cpu_pct": null,
  "train_rss_gb": null,
  "throttle_state": "$throttle_state",
  "throttle_note": "training ended"
}
EOF
        log "training ended, exiting"
        exit 0
    fi

    sleep 10
done
