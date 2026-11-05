#!/usr/bin/env bash
# hjepa-run.sh: run an H-JEPA planning evaluation (eval.py of the official release) on
#   local   this computer: an NVIDIA GPU directly when there is one, otherwise the Apple GPU (MPS)
#           or the CPU through hjepa_local.py;
#   remote  a remote NVIDIA GPU machine over SSH: eval.py runs there unchanged and the results are
#           copied back.
# With --setup, whatever the run needs and is missing (code, Python environment, checkpoint,
# evaluation tasks or DROID clips) is installed first with install-hjepa.sh, pinned.
#
# Simulated environments (ant, fourroom, cube, pusht) report success_rate (% of the 50 tasks
# reached). DROID reports Frechet fidelity (% match of the planned to the recorded robot path,
# over 16 clips, 0 = no motion, 100 = exact).
#
# Results are copied to <results>/<run-id>/ with a run.json summary; <record> gets a copy of it.
# Shipped by the World Model (H-JEPA) PI-Desktop plugin. Bash 3.2+ (macOS) and Linux.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET=local
HOST=""
REMOTE_HOME="hjepa"
LOCAL_HOME="${HJEPA_HOME:-$HOME/hjepa}"
CODE_DIR=""
DEVICE=auto
ENV_NAME=ant
MODEL=hjepa_l3
SEED=""
PLAN_SEED=1
TASKS=""
OUT=""
SETUP=0
PYMODE=auto
DRY=0
RESULTS=".world-model/hjepa/results"
RECORD=".world-model/hjepa/last-run.json"
EXTRA=()

usage() {
  cat <<'EOF'
Usage: hjepa-run.sh [options] [-- extra hydra overrides]

What to evaluate:
  --env ENV          ant | fourroom | cube | pusht | droid           [ant]
  --model MODEL      lewm | hjepa_l2..l4 | hwm_l2..l4                [hjepa_l3]
                     (droid: lewm | hjepa_l2 | hwm_l2)
  --seed SEED        training seed: 42 | 43 | 44 (droid: 1 | 1000 | 10000)   [42; droid: 1]
  --plan-seed N      droid only: planner seed (the paper uses 1, 2, 3) [1]
  --tasks N          the first N tasks (simulated: of 50) or clips (droid: of 16)   [all]
  --out NAME         output.dir under the checkpoint folder          [eval_pi_<target>[_n<N>]]

Where:
  --target T         local | remote                                  [local]
  --device D         local only: auto | mps | cpu                    [auto: mps if available]
  --home DIR         local HJEPA_HOME                                [$HJEPA_HOME or ~/hjepa]
  --code-dir DIR     local code clone                                [<home>/H-JEPA]
  --host HOST        remote: SSH destination (user@host or an ~/.ssh/config alias, key auth)
  --remote-home DIR  remote HJEPA_HOME, relative to the remote home  [hjepa]
  --setup            install what the run needs and is missing first (install-hjepa.sh, pinned)
  --python MODE      for --setup: auto | conda | uv | venv           [auto]

Output:
  --results DIR      where results are copied                        [.world-model/hjepa/results]
  --record FILE      copy of the run summary                         [.world-model/hjepa/last-run.json]
  --dry-run          print the commands without running them
  -h, --help

A run is resumable: rerunning with the same --out skips the tasks or clips already done.
EOF
}

die() { printf 'hjepa-run: error: %s\n' "$*" >&2; exit 2; }
note() { printf 'hjepa-run: %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="${2:?}"; shift 2 ;;
    --host) HOST="${2:?}"; shift 2 ;;
    --remote-home) REMOTE_HOME="${2:?}"; shift 2 ;;
    --home) LOCAL_HOME="${2:?}"; shift 2 ;;
    --code-dir) CODE_DIR="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --env) ENV_NAME="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --plan-seed) PLAN_SEED="${2:?}"; shift 2 ;;
    --tasks) TASKS="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --setup) SETUP=1; shift ;;
    --python) PYMODE="${2:?}"; shift 2 ;;
    --results) RESULTS="${2:?}"; shift 2 ;;
    --record) RECORD="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

# ---------- validation (values end up in shell commands, locally and over SSH) ----------
case "$TARGET" in local|remote) ;; *) die "--target must be local or remote" ;; esac
case "$DEVICE" in auto|mps|cpu) ;; *) die "--device must be auto, mps or cpu" ;; esac
case "$PYMODE" in auto|conda|uv|venv) ;; *) die "--python must be auto, conda, uv or venv" ;; esac
if [ "$ENV_NAME" = droid ]; then
  [ -n "$SEED" ] || SEED=1
  case "$MODEL" in lewm|hjepa_l2|hwm_l2) ;; *) die "model '$MODEL' does not exist for droid (lewm, hjepa_l2, hwm_l2)" ;; esac
  case "$SEED" in 1|1000|10000) ;; *) die "seed '$SEED' does not exist for droid (1, 1000, 10000)" ;; esac
  case "$PLAN_SEED" in *[!0-9]*|"") die "--plan-seed must be a number" ;; esac
  MAX_TASKS=16
  UNIT=clip
else
  [ -n "$SEED" ] || SEED=42
  case "$ENV_NAME" in ant|fourroom|cube|pusht) ;; *) die "unknown environment '$ENV_NAME' (ant, fourroom, cube, pusht, droid)" ;; esac
  case "$MODEL" in lewm|hjepa_l2|hjepa_l3|hjepa_l4|hwm_l2|hwm_l3|hwm_l4) ;; *) die "model '$MODEL' does not exist for $ENV_NAME" ;; esac
  case "$SEED" in 42|43|44) ;; *) die "seed '$SEED' does not exist for $ENV_NAME (42, 43, 44)" ;; esac
  MAX_TASKS=50
  UNIT=task
fi
if [ -n "$TASKS" ]; then
  case "$TASKS" in *[!0-9]*|0) die "--tasks must be a number from 1 to $MAX_TASKS" ;; esac
  [ "$TASKS" -le "$MAX_TASKS" ] || die "--tasks must be a number from 1 to $MAX_TASKS"
fi
[ -n "$OUT" ] || OUT="eval_pi_${TARGET}${TASKS:+_n$TASKS}"
case "$OUT" in *[!A-Za-z0-9_.-]*|.*) die "--out may only contain letters, digits, '_', '-', '.'" ;; esac
if [ "$TARGET" = remote ]; then
  [ -n "$HOST" ] || die "--host is required for --target remote (set the plugin's 'Remote GPU host' setting)"
  case "$HOST" in -*|*[!A-Za-z0-9_.@:-]*) die "--host may only contain letters, digits and _ . @ : -" ;; esac
  REMOTE_HOME="${REMOTE_HOME#\~/}"
  case "$REMOTE_HOME" in ""|-*|*[!A-Za-z0-9_./-]*|*..*) die "--remote-home must be a plain path (letters, digits, _ . / -), relative to the remote home or absolute" ;; esac
fi
for x in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$x" in *[!A-Za-z0-9_.=+/:,-]*) die "extra override '$x' contains unsupported characters" ;; esac
done

case "$MODEL" in lewm) PLANNER=flat ;; *) PLANNER="${MODEL##*_}" ;; esac
CFG="${ENV_NAME}_${PLANNER}"
CK_REL="ckpts/$ENV_NAME/${ENV_NAME}_$MODEL/seed$SEED"
CK_FILE="${ENV_NAME}_${MODEL}_object.ckpt"
if [ "$ENV_NAME" = droid ]; then
  # The DROID config's seed is the planner seed (scripts/eval_droid.sh: 1, 2, 3); num_eval is top level.
  HYDRA=(--config-name "$CFG" "seed=$PLAN_SEED" "output.dir=$OUT")
  [ -z "$TASKS" ] || HYDRA+=("num_eval=$TASKS")
  RESULT_FILES="eval.csv plan_config.yaml"
  RUN_ID="${TARGET}-${ENV_NAME}_${MODEL}-s${SEED}-p${PLAN_SEED}-${OUT}"
else
  # Simulated environments: the planner seed is the model seed, as in the paper.
  HYDRA=(--config-name "$CFG" "seed=$SEED" "output.dir=$OUT")
  [ -z "$TASKS" ] || HYDRA+=("eval.num_eval=$TASKS")
  RESULT_FILES="metrics.yaml planning_compute.json eval_config.yaml"
  RUN_ID="${TARGET}-${ENV_NAME}_${MODEL}-s${SEED}-${OUT}"
fi
case "$RESULTS" in /*) ;; *) RESULTS="$PWD/$RESULTS" ;; esac
case "$RECORD" in /*) ;; *) RECORD="$PWD/$RECORD" ;; esac
DEST="$RESULTS/$RUN_ID"

run() {
  { printf '+'; printf ' %s' "$@"; printf '\n'; } >&2
  [ "$DRY" = 1 ] || "$@"
}
sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
sqall() { local a out=""; for a in "$@"; do out="$out $(sq "$a")"; done; printf '%s' "${out# }"; }
json_str() { printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }
num_or_null() { case "$1" in ""|*[!0-9.eE+-]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }
yaml_metric() { [ -f "$2" ] || return 0; sed -n "s/^$1: *//p" "$2" | head -1; }
csv_metric() { # column file: value of that column in the 2-line eval.csv
  [ -f "$2" ] || return 0
  awk -F, -v col="$1" '{ sub(/\r$/, "") } NR==1 { for (i = 1; i <= NF; i++) if ($i == col) c = i } NR==2 && c { print $c }' "$2"
}

STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_S="$(date +%s)"
DEVICE_USED=""
INSTALLED=""

# Score of a finished run: "<metric> <value-in-percent> <count>"
score() {
  local m v n
  if [ "$ENV_NAME" = droid ]; then
    v="$(csv_metric frechet/skill_mean "$DEST/eval.csv")"
    n="$(csv_metric n_episodes "$DEST/eval.csv")"
    [ -z "$v" ] || v="$(awk -v x="$v" 'BEGIN { printf "%.2f", 100 * x }')"
    m=frechet_fidelity
  else
    v="$(yaml_metric success_rate "$DEST/metrics.yaml")"
    n="${TASKS:-50}"
    m=success_rate
  fi
  printf '%s %s %s' "$m" "${v:-}" "${n:-}"
}

record() { # status
  [ "$DRY" = 0 ] || return 0
  mkdir -p "$DEST" "$(dirname "$RECORD")"
  local s m v n et
  s="$(score)"; m="${s%% *}"; s="${s#* }"; v="${s%% *}"; n="${s#* }"
  et="$(yaml_metric evaluation_time "$DEST/metrics.yaml")"
  {
    printf '{"status":%s,"runId":%s,"target":%s,' "$(json_str "$1")" "$(json_str "$RUN_ID")" "$(json_str "$TARGET")"
    printf '"host":%s,"device":%s,"installed":%s,' "$(json_str "$HOST")" "$(json_str "$DEVICE_USED")" "$(json_str "$INSTALLED")"
    printf '"env":%s,"model":%s,"seed":%s,"config":%s,"out":%s,' "$(json_str "$ENV_NAME")" "$(json_str "$MODEL")" "$(json_str "$SEED")" "$(json_str "$CFG")" "$(json_str "$OUT")"
    [ "$ENV_NAME" != droid ] || printf '"planSeed":%s,' "$PLAN_SEED"
    printf '"unit":%s,"count":%s,"metric":%s,"score":%s,' "$(json_str "$UNIT")" "$(num_or_null "${n:-${TASKS:-$MAX_TASKS}}")" "$(json_str "$m")" "$(num_or_null "$v")"
    if [ "$m" = success_rate ]; then printf '"successRate":%s,' "$(num_or_null "$v")"; else printf '"frechetFidelity":%s,' "$(num_or_null "$v")"; fi
    printf '"evaluationTime":%s,"wallSeconds":%s,' "$(num_or_null "$et")" "$(( $(date +%s) - START_S ))"
    printf '"started":%s,"finished":%s,"results":%s}\n' "$(json_str "$STARTED")" "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" "$(json_str "$DEST")"
  } > "$DEST/run.json"
  cp "$DEST/run.json" "$RECORD"
}

# What a run needs, as one shell snippet that prints the missing parts (run in HJEPA_HOME).
# Arguments: the code dir and the python path, both relative to HJEPA_HOME or absolute.
needs_check() {
  local code="$1" py="$2" assets
  if [ "$ENV_NAME" = droid ]; then
    assets='[ -d droid/droid_raw ] || echo droid-clips; '
  else
    local d; case "$ENV_NAME" in cube) d=ogbench ;; *) d="$ENV_NAME" ;; esac
    assets="ls $(sq "$code")/h_jepa/assets/eval_trajs/$d/*.pt >/dev/null 2>&1 || echo eval-tasks; "
  fi
  printf '%s' "[ -d $(sq "$code")/h_jepa ] || echo code; [ -x $(sq "$py") ] || echo python; "
  printf '%s' "[ -s $(sq "$CK_REL/$CK_FILE") ] && [ -s $(sq "$CK_REL/normalizer.pt") ] || echo checkpoint; $assets"
  [ "$ENV_NAME" != droid ] || printf '%s' "[ -x $(sq "$py") ] && $(sq "$py") -c 'import decord' >/dev/null 2>&1 || echo decord; "
}

INSTALL_FLAGS=(--env "$ENV_NAME" --models "$MODEL" --seeds "$SEED" --python "$PYMODE")

# ---------- local ----------
run_local() {
  case "$LOCAL_HOME" in /*) ;; *) LOCAL_HOME="$PWD/$LOCAL_HOME" ;; esac
  local code="${CODE_DIR:-$LOCAL_HOME/H-JEPA}" py="$LOCAL_HOME/env/bin/python"
  local ckpt="$LOCAL_HOME/$CK_REL/$CK_FILE" outdir="$LOCAL_HOME/$CK_REL/$OUT" missing=""
  missing="$( (cd "$LOCAL_HOME" 2>/dev/null && /bin/bash -c "$(needs_check "$code" "$py")") || echo code python checkpoint)"
  missing="$(printf '%s' "$missing" | tr '\n' ' ' | sed 's/ *$//')"
  if [ -n "$missing" ]; then
    if [ "$SETUP" = 1 ]; then
      note "missing locally: $missing; installing with install-hjepa.sh"
      [ -f "$SCRIPT_DIR/install-hjepa.sh" ] || die "install-hjepa.sh is missing next to this script"
      INSTALLED="$missing"
      run bash "$SCRIPT_DIR/install-hjepa.sh" --home "$LOCAL_HOME" ${CODE_DIR:+--code-dir "$CODE_DIR"} "${INSTALL_FLAGS[@]}" --state "$(dirname "$RECORD")/state.json"
    elif [ "$DRY" = 0 ]; then
      die "missing in $LOCAL_HOME: $missing. Rerun with --setup, or: install-hjepa.sh --env $ENV_NAME --models $MODEL --seeds $SEED"
    fi
  fi
  local launcher=() gl="${MUJOCO_GL:-}"
  if [ "$(uname -s)" = Linux ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DEVICE_USED=cuda
    note "NVIDIA GPU found: running eval.py unchanged"
  else
    [ -f "$SCRIPT_DIR/hjepa_local.py" ] || die "hjepa_local.py is missing next to this script"
    launcher=("$SCRIPT_DIR/hjepa_local.py")
    DEVICE_USED="$DEVICE"
    if [ "$DEVICE" = auto ] && [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then DEVICE_USED=mps; fi
    [ "$(uname -s)" != Darwin ] || gl="${gl:-glfw}"   # eval.py forces egl, which macOS lacks
    note "no NVIDIA GPU: running through hjepa_local.py on $DEVICE_USED (results are close to, not identical with, CUDA runs)"
  fi
  printf '+ cd %s\n' "$code/h_jepa" >&2
  if [ "$DRY" = 0 ]; then cd "$code/h_jepa"; fi
  run env "HJEPA_HOME=$LOCAL_HOME" "HJEPA_DEVICE=$DEVICE" ${gl:+"MUJOCO_GL=$gl"} PYTHONUNBUFFERED=1 "$py" ${launcher[@]+"${launcher[@]}"} eval.py \
    "${HYDRA[@]}" "policy=$ckpt" ${EXTRA[@]+"${EXTRA[@]}"}
  cd - >/dev/null 2>&1 || true
  run mkdir -p "$DEST"
  local f
  for f in $RESULT_FILES; do
    if [ "$DRY" = 1 ] || [ -f "$outdir/$f" ]; then run cp "$outdir/$f" "$DEST/$f"; fi
  done
}

# ---------- remote ----------
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=60)

run_remote() {
  local rh; rh="$(sq "$REMOTE_HOME")"
  printf '+ preflight: ssh %s (Linux, nvidia-smi, what is installed)\n' "$HOST" >&2
  local missing="code python checkpoint"
  if [ "$DRY" = 0 ]; then
    local pre
    pre="$(ssh "${SSH_OPTS[@]}" "$HOST" "uname -s; if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null; then :; else echo NO_NVIDIA_GPU; fi; echo MISSING: \$(cd $rh 2>/dev/null && { $(needs_check H-JEPA env/bin/python) } || echo code python checkpoint)")" \
      || die "cannot reach $HOST over SSH without a password prompt: check the host name and your SSH key (ssh $HOST)"
    printf '%s\n' "$pre" | grep -v '^MISSING:' | sed 's/^/  remote: /' >&2
    case "$pre" in *NO_NVIDIA_GPU*) die "$HOST has no working nvidia-smi: it is not an NVIDIA GPU machine" ;; esac
    DEVICE_USED="cuda ($(printf '%s\n' "$pre" | sed -n 's/^GPU 0: \([^(]*\).*/\1/p' | sed 's/ *$//'))"
    missing="$(printf '%s\n' "$pre" | sed -n 's/^MISSING: *//p' | sed 's/ *$//')"
  else
    DEVICE_USED=cuda
  fi
  if [ -n "$missing" ]; then
    [ "$SETUP" = 1 ] || die "missing on $HOST: $missing. Rerun with --setup to install it there"
    note "missing on $HOST: $missing; installing there with install-hjepa.sh"
    [ -f "$SCRIPT_DIR/install-hjepa.sh" ] || die "install-hjepa.sh is missing next to this script"
    INSTALLED="$missing"
    run ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $rh/.pi"
    run scp -q "${SSH_OPTS[@]}" "$SCRIPT_DIR/install-hjepa.sh" "$HOST:$REMOTE_HOME/.pi/install-hjepa.sh"
    run ssh "${SSH_OPTS[@]}" "$HOST" "bash $rh/.pi/install-hjepa.sh --home $rh $(sqall "${INSTALL_FLAGS[@]}") --state $rh/.pi/state.json"
  fi
  local cmd
  cmd="cd $rh && H=\$(pwd) && test -x \"\$H/env/bin/python\" && test -s \"\$H/$CK_REL/$CK_FILE\" || { echo 'hjepa-run: release or checkpoint missing on the remote host: rerun with --setup' >&2; exit 3; }; cd \"\$H/H-JEPA/h_jepa\" && HJEPA_HOME=\"\$H\" MUJOCO_GL=egl PYTHONUNBUFFERED=1 \"\$H/env/bin/python\" eval.py $(sqall "${HYDRA[@]}") \"policy=\$H/$CK_REL/$CK_FILE\""
  [ ${#EXTRA[@]} -eq 0 ] || cmd="$cmd $(sqall "${EXTRA[@]}")"
  run ssh "${SSH_OPTS[@]}" "$HOST" "$cmd"
  run rm -rf "$DEST"
  run mkdir -p "$RESULTS"
  run scp -q -r "${SSH_OPTS[@]}" "$HOST:$REMOTE_HOME/$CK_REL/$OUT" "$DEST"
}

note "run $RUN_ID: $CFG on ${TARGET}${HOST:+ ($HOST)}, ${TASKS:-all $MAX_TASKS} ${UNIT}(s), output.dir=$OUT"
trap 'rc=$?; [ $rc -eq 0 ] || record failed; exit $rc' EXIT
if [ "$TARGET" = local ]; then run_local; else run_remote; fi
if [ "$DRY" = 1 ]; then note "dry run: nothing was changed"; trap - EXIT; exit 0; fi
record done
trap - EXIT
s="$(score)"; m="${s%% *}"; s="${s#* }"
note "done: ${m} ${s%% *}% over ${s#* } ${UNIT}(s); results in $DEST"
