# Every managed child is launched with setsid so its descendants can be stopped
# without signaling another job or this batch shell's own process group.
SERVER_PID= PROXY_PID= WATCH_PID= STAGE_PID=

cleanup_services() {
  local status=$?
  trap - EXIT TERM INT
  local groups=() pid attempt alive
  for pid in "$STAGE_PID" "$WATCH_PID" "$PROXY_PID" "$SERVER_PID"; do
    [[ -n "$pid" ]] && groups+=("-$pid")
  done
  if ((${#groups[@]})); then
    kill -TERM -- "${groups[@]}" 2>/dev/null || true
    for ((attempt=0; attempt<10; attempt++)); do
      alive=0
      for pid in "${groups[@]}"; do
        kill -0 -- "$pid" 2>/dev/null && alive=1
      done
      ((alive)) || break
      sleep 1
    done
    kill -KILL -- "${groups[@]}" 2>/dev/null || true
  fi
  # Never use unbounded wait here: Slurm reaps remaining processes at job exit.
  exit "$status"
}

run_logged() {
  local log=$1
  shift
  setsid bash -c 'set -o pipefail; log=$1; shift; "$@" 2>&1 | tee "$log"' _ "$log" "$@" &
  STAGE_PID=$!
  # Bash's builtin wait is interruptible, unlike a foreground pipeline. A
  # watchdog signal therefore enters cleanup immediately and stops the stage.
  wait "$STAGE_PID"
  STAGE_PID=
}

trap cleanup_services EXIT
trap 'exit 1' TERM INT
