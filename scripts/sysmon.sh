#!/usr/bin/env bash
# CP5: proactive throttle + resource monitor (post-reboot)
# Loops every 10s. Writes docs/sysstats.json each tick.
# Proactively applies background-QoS+renice+5 on first pid discovery.
# Escalates to renice+10 after 3 consecutive critical-mem ticks.
# Exits after 5 consecutive empty-pid ticks.

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STATS_FILE="$PROJECT_ROOT/docs/sysstats.json"
LOG_FILE="$SCRIPT_DIR/sysmon.log"

critical_ticks=0    # consecutive ticks with mem_pressure==critical
empty_ticks=0       # consecutive ticks with no training pid
throttle_state="none"
throttle_note=""
proactive_done=0    # 1 once initial background-qos+renice5 applied
escalated=0         # 1 once renice+10 escalation applied

log() {
    echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG_FILE"
}

write_json() {
    local ts="$1"
    local total_ram_gb="$2"
    local mem_free_pct="$3"
    local mem_pressure="$4"
    local mem_used_gb="$5"
    local swap_used_gb="$6"
    local cpu_speed_limit_pct="$7"
    local load_avg="$8"
    local pid_json="$9"
    local train_cpu_pct="${10}"
    local train_rss_gb="${11}"
    local t_state="${12}"
    local t_note="${13}"

    if [[ -z "$t_note" ]]; then
        t_note_json="null"
    else
        # escape any double-quotes in the note
        escaped_note="${t_note//\"/\\\"}"
        t_note_json="\"${escaped_note}\""
    fi

    cat > "$STATS_FILE" <<ENDJSON
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
  "throttle_state": "$t_state",
  "throttle_note": $t_note_json
}
ENDJSON
}

log "sysmon started pid=$$"

while true; do
    ts=$(date -u +%H:%M:%SZ)

    # --- RAM ---
    total_ram_gb=$(( $(sysctl -n hw.memsize) / 1000000000 ))

    # --- Memory free % ---
    mem_free_pct=$(memory_pressure 2>/dev/null \
        | grep -i "System-wide memory free percentage" \
        | grep -oE '[0-9]+' | head -1)
    [[ -z "$mem_free_pct" ]] && mem_free_pct=0

    if (( mem_free_pct < 10 )); then
        mem_pressure="critical"
    elif (( mem_free_pct < 25 )); then
        mem_pressure="warn"
    else
        mem_pressure="normal"
    fi

    mem_used_gb=$(awk "BEGIN {printf \"%.2f\", $total_ram_gb * (1 - $mem_free_pct/100)}")

    # --- Swap ---
    swap_raw=$(sysctl -n vm.swapusage 2>/dev/null || echo "")
    swap_used_gb="null"
    if [[ -n "$swap_raw" ]]; then
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
        [[ -n "$limit" ]] && cpu_speed_limit_pct="$limit"
    fi

    # --- Load ---
    load_avg=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
    [[ -z "$load_avg" ]] && load_avg="null"

    # --- Training pid ---
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

        # --- B) PROACTIVE THROTTLE: apply immediately on first discovery ---
        if (( proactive_done == 0 )); then
            taskpolicy -b -p "$pid" 2>/dev/null \
                && log "taskpolicy -b applied proactively to pid=$pid" \
                || log "taskpolicy -b FAILED for pid=$pid"
            renice +5 -p "$pid" 2>/dev/null \
                && log "renice +5 applied proactively to pid=$pid" \
                || log "renice +5 FAILED for pid=$pid"
            throttle_state="background-qos+renice (proactive)"
            throttle_note="applied @${ts} to keep system responsive"
            log "proactive throttle: $throttle_note"
            proactive_done=1
        fi

        # --- C) ESCALATE after 3 consecutive critical ticks ---
        if [[ "$mem_pressure" == "critical" ]]; then
            critical_ticks=$(( critical_ticks + 1 ))
        else
            critical_ticks=0
        fi

        if (( critical_ticks >= 3 && escalated == 0 )); then
            renice +10 -p "$pid" 2>/dev/null \
                && log "renice +10 ESCALATION applied to pid=$pid (critical_ticks=$critical_ticks)" \
                || log "renice +10 ESCALATION FAILED for pid=$pid"
            throttle_state="background-qos+renice (escalated)"
            throttle_note="ESCALATED @${ts}: critical mem pressure (${critical_ticks} consecutive)"
            log "escalation: $throttle_note"
            escalated=1
        fi

    else
        empty_ticks=$(( empty_ticks + 1 ))
        critical_ticks=0
    fi

    # --- D) EXIT when training gone for 5 consecutive ticks ---
    if (( empty_ticks >= 5 )); then
        final_note="training ended @${ts}"
        write_json "$ts" "$total_ram_gb" "$mem_free_pct" "$mem_pressure" \
            "$mem_used_gb" "$swap_used_gb" "$cpu_speed_limit_pct" "$load_avg" \
            "null" "null" "null" "$throttle_state" "$final_note"
        log "training ended, sysmon exiting"
        exit 0
    fi

    write_json "$ts" "$total_ram_gb" "$mem_free_pct" "$mem_pressure" \
        "$mem_used_gb" "$swap_used_gb" "$cpu_speed_limit_pct" "$load_avg" \
        "$pid_json" "$train_cpu_pct" "$train_rss_gb" \
        "$throttle_state" "$throttle_note"

    sleep 10
done
