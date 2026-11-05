#!/usr/bin/env bash
# install-hjepa.sh: set up the official H-JEPA release.
#
#   code         https://github.com/kevinghst/H-JEPA           (MIT, pinned commit)
#   checkpoints  https://huggingface.co/jepa-world-models/h-jepa (MIT, pinned revision)
#   eval tasks   https://huggingface.co/datasets/jepa-world-models/h-jepa (CC BY 4.0, pinned)
#
# It follows the release's own README (sections 1, 3.2, 3.4, 4.1, 4.3):
#   - clone the code at a pinned commit;
#   - create a Python 3.11 environment;
#   - pip install -e ".[train,env]";
#   - download the selected checkpoints into $HJEPA_HOME/ckpts and the evaluation tasks into
#     h_jepa/assets/eval_trajs, verify their checksums;
#   - load one checkpoint as a smoke test.
# Training datasets (tens of GB) are out of scope: see the release README, section 3.1 / 4.1.
#
# Shipped by the World Model (H-JEPA) PI-Desktop plugin. Bash 3.2+ (macOS) and Linux.

set -euo pipefail

CODE_REPO="https://github.com/kevinghst/H-JEPA.git"
CODE_REV="97054ecaff83e13839195ec4b92d74c08a0127c5"
MODEL_REPO="jepa-world-models/h-jepa"
MODEL_REV="08fc8a21208671a87741f2960aeba9b5db4420e0"
DATA_REPO="jepa-world-models/h-jepa"
DATA_REV="cc05cb8d2aaf9aec0d5364d073c6369ff760b60b"
PYTHON_VERSION="3.11.13"

HOME_DIR="${HJEPA_HOME:-$HOME/hjepa}"
CODE_DIR=""
ENVS="ant"
MODELS="default"
SEEDS="default"
EXTRAS="auto"
PYMODE="auto"
DRY=0
CHECK=0
STATE=""
SKIP_CODE=0
SKIP_TASKS=0
SKIP_SMOKE=0

usage() {
  cat <<'EOF'
Usage: install-hjepa.sh [options]

Selection (comma-separated):
  --env ENVS       ant | fourroom | cube | pusht | droid | sim (the 4 simulated) | all   [ant]
  --models MODELS  lewm, hjepa_l2, hjepa_l3, hjepa_l4, hwm_l2, hwm_l3, hwm_l4 | all
                   (droid: lewm, hjepa_l2, hwm_l2)       [hjepa_l3; droid: hjepa_l2]
  --seeds SEEDS    42, 43, 44 (droid: 1, 1000, 10000) | all          [42; droid: 1]

Locations:
  --home DIR       data root, exported as HJEPA_HOME             [$HJEPA_HOME or ~/hjepa]
  --code-dir DIR   where the code is cloned                       [<home>/H-JEPA]

Environment:
  --python MODE    auto | conda | uv | venv                       [auto: conda, uv, python3.11]
  --extras LIST    pip extras of the code release                 [auto: train,env]

Steps:
  --skip-code      only download (code and environment already set up)
  --skip-tasks     do not download the evaluation tasks
  --skip-smoke     do not load a checkpoint at the end
  --check          only report what is installed; exit 1 if something is missing
  --state FILE     write a JSON summary to FILE (used by the plugin's wm_hjepa tool)
  --dry-run        print the commands without running them
  -h, --help
EOF
}

die() { printf 'install-hjepa: error: %s\n' "$*" >&2; exit 2; }
note() { printf 'install-hjepa: %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --env|--envs) ENVS="${2:?}"; shift 2 ;;
    --models) MODELS="${2:?}"; shift 2 ;;
    --seeds) SEEDS="${2:?}"; shift 2 ;;
    --home) HOME_DIR="${2:?}"; shift 2 ;;
    --code-dir) CODE_DIR="${2:?}"; shift 2 ;;
    --python) PYMODE="${2:?}"; shift 2 ;;
    --extras) EXTRAS="${2:?}"; shift 2 ;;
    --skip-code) SKIP_CODE=1; shift ;;
    --skip-tasks) SKIP_TASKS=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --check) CHECK=1; shift ;;
    --state) STATE="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

case "$HOME_DIR" in /*) ;; *) HOME_DIR="$PWD/$HOME_DIR" ;; esac
[ -n "$CODE_DIR" ] || CODE_DIR="$HOME_DIR/H-JEPA"
case "$CODE_DIR" in /*) ;; *) CODE_DIR="$PWD/$CODE_DIR" ;; esac
ENV_DIR="$HOME_DIR/env"
CKPT_DIR="$HOME_DIR/ckpts"
TASK_DIR="$CODE_DIR/h_jepa/assets"

# ---------- catalog (sizes in MB per checkpoint run, from the pinned revision) ----------
SIM_ENVS="ant fourroom cube pusht"
SIM_MODELS="lewm hjepa_l2 hjepa_l3 hjepa_l4 hwm_l2 hwm_l3 hwm_l4"
SIM_SEEDS="42 43 44"
DROID_MODELS="lewm hjepa_l2 hwm_l2"
DROID_SEEDS="1 1000 10000"

run_mb() {
  case "$1/$2" in
    ant/lewm) echo 50 ;; ant/hjepa_l2) echo 95 ;; ant/hjepa_l3) echo 140 ;; ant/hjepa_l4) echo 185 ;;
    ant/hwm_l2) echo 93 ;; ant/hwm_l3) echo 137 ;; ant/hwm_l4) echo 181 ;;
    fourroom/lewm) echo 59 ;; fourroom/hjepa_l2) echo 66 ;; fourroom/hjepa_l3) echo 87 ;; fourroom/hjepa_l4) echo 108 ;;
    fourroom/hwm_l2) echo 66 ;; fourroom/hwm_l3) echo 86 ;; fourroom/hwm_l4) echo 106 ;;
    cube/lewm|pusht/lewm) echo 45 ;; cube/hjepa_l2|pusht/hjepa_l2|cube/hwm_l2|pusht/hwm_l2) echo 93 ;;
    cube/hjepa_l3|pusht/hjepa_l3) echo 141 ;; cube/hwm_l3|pusht/hwm_l3) echo 140 ;;
    cube/hjepa_l4|pusht/hjepa_l4) echo 189 ;; cube/hwm_l4|pusht/hwm_l4) echo 187 ;;
    droid/lewm) echo 216 ;; droid/hjepa_l2) echo 348 ;; droid/hwm_l2) echo 345 ;;
    *) echo 0 ;;
  esac
}

# evaluation tasks per environment: dataset path and size in MB
task_path() {
  case "$1" in
    ant) echo "eval_trajs/ant/*" ;; fourroom) echo "eval_trajs/fourroom/*" ;;
    cube) echo "eval_trajs/ogbench/*" ;; pusht) echo "eval_trajs/pusht/*" ;;
    droid) echo "droid/droid_raw_eval16.tar" ;;
  esac
}
task_mb() { case "$1" in ant) echo 155 ;; fourroom) echo 574 ;; cube) echo 152 ;; pusht) echo 565 ;; droid) echo 252 ;; esac; }

contains() { case " $1 " in *" $2 "*) return 0 ;; *) return 1 ;; esac; }
commas() { printf '%s' "$1" | tr ',' ' '; }

# ---------- expand and validate the selection ----------
case "$ENVS" in
  all) SEL_ENVS="$SIM_ENVS droid" ;;
  sim) SEL_ENVS="$SIM_ENVS" ;;
  *) SEL_ENVS="$(commas "$ENVS")" ;;
esac
RUNS=""   # "env/model/seed" items
TOTAL_MB=0
for e in $SEL_ENVS; do
  if [ "$e" = droid ]; then all_m="$DROID_MODELS"; all_s="$DROID_SEEDS"; def_m=hjepa_l2; def_s=1
  elif contains "$SIM_ENVS" "$e"; then all_m="$SIM_MODELS"; all_s="$SIM_SEEDS"; def_m=hjepa_l3; def_s=42
  else die "unknown environment '$e' (ant, fourroom, cube, pusht, droid, sim, all)"; fi
  case "$MODELS" in all) ms="$all_m" ;; default) ms="$def_m" ;; *) ms="$(commas "$MODELS")" ;; esac
  case "$SEEDS" in all) ss="$all_s" ;; default) ss="$def_s" ;; *) ss="$(commas "$SEEDS")" ;; esac
  for m in $ms; do
    contains "$all_m" "$m" || die "model '$m' does not exist for $e (have: $all_m)"
    for s in $ss; do
      contains "$all_s" "$s" || die "seed '$s' does not exist for $e (have: $all_s)"
      RUNS="$RUNS $e/$m/$s"
      TOTAL_MB=$((TOTAL_MB + $(run_mb "$e" "$m")))
    done
  done
  [ "$SKIP_TASKS" = 1 ] || TOTAL_MB=$((TOTAL_MB + $(task_mb "$e")))
done
RUNS="${RUNS# }"
[ -n "$RUNS" ] || die "nothing selected"

ckpt_file() { # env/model/seed -> path of the _object.ckpt
  local e="${1%%/*}" rest="${1#*/}"; local m="${rest%%/*}" s="${rest#*/}"
  printf '%s/%s/%s_%s/seed%s/%s_%s_object.ckpt' "$CKPT_DIR" "$e" "$e" "$m" "$s" "$e" "$m"
}

run() {
  { printf '+'; printf ' %s' "$@"; printf '\n'; } >&2
  [ "$DRY" = 1 ] || "$@"
}

sha_check() { # dir file
  if command -v sha256sum >/dev/null 2>&1; then (cd "$1" && sha256sum -c --ignore-missing "$2")
  else (cd "$1" && shasum -a 256 -c --ignore-missing "$2"); fi
}

json_str() { printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }

# ---------- python environment ----------
PY="$ENV_DIR/bin/python"
PIP_KIND=""
detect_env_kind() {
  if [ -f "$ENV_DIR/.hjepa-kind" ]; then PIP_KIND="$(cat "$ENV_DIR/.hjepa-kind")"; return; fi
  case "$PYMODE" in
    conda|uv|venv) PIP_KIND="$PYMODE" ;;
    auto)
      if command -v conda >/dev/null 2>&1; then PIP_KIND=conda
      elif command -v uv >/dev/null 2>&1; then PIP_KIND=uv
      elif command -v python3.11 >/dev/null 2>&1; then PIP_KIND=venv
      elif [ "$DRY" = 1 ]; then PIP_KIND=uv; note "dry run: no conda, uv or python3.11 found; showing the uv commands (a real run stops here)"
      else die "no conda, uv or python3.11 found. Install one (e.g. 'brew install uv' or Miniforge), or pass --python"; fi ;;
    *) die "--python must be auto, conda, uv or venv" ;;
  esac
}

pip_install() {
  if [ "$PIP_KIND" = uv ]; then run uv pip install --python "$PY" "$@"
  else run "$PY" -m pip install "$@"; fi
}

torch_index_args() {
  # The release installs CUDA 12.8 wheels. Only use that index where it can help (Linux + NVIDIA).
  if [ "$(uname -s)" = Linux ] && command -v nvidia-smi >/dev/null 2>&1; then
    if [ "$PIP_KIND" = uv ]; then echo "--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match"
    else echo "--extra-index-url https://download.pytorch.org/whl/cu128"; fi
  fi
}

install_code() {
  if [ -d "$CODE_DIR/.git" ]; then
    note "code already cloned at $CODE_DIR"
    run git -C "$CODE_DIR" fetch --quiet origin "$CODE_REV" || true
  else
    run mkdir -p "$(dirname "$CODE_DIR")"
    run git clone --quiet "$CODE_REPO" "$CODE_DIR"
  fi
  run git -C "$CODE_DIR" -c advice.detachedHead=false checkout --quiet "$CODE_REV"

  detect_env_kind
  if [ ! -x "$PY" ]; then
    run mkdir -p "$HOME_DIR"
    case "$PIP_KIND" in
      conda) run conda create -y -q -p "$ENV_DIR" "python=$PYTHON_VERSION" ;;
      uv) run uv venv -q --python "$PYTHON_VERSION" "$ENV_DIR" ;;
      venv) run python3.11 -m venv "$ENV_DIR" ;;
    esac
    [ "$DRY" = 1 ] || printf '%s\n' "$PIP_KIND" > "$ENV_DIR/.hjepa-kind"
  fi
  [ "$PIP_KIND" = uv ] || pip_install -q --upgrade pip

  local extras="$EXTRAS" idx
  idx="$(torch_index_args)"
  if [ "$extras" = auto ] && [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    # decord (a 'train' extra, the video reader DROID needs) has no macOS arm64 wheel. Install the
    # other 'train' dependencies from the pinned pyproject.toml, then a maintained fork that ships
    # arm64 wheels and imports as 'decord': decord2 (macOS 14+), else eva-decord (macOS 11+).
    note "macOS arm64: installing [env], the [train] dependencies except decord, and decord2 (or eva-decord) in its place"
    # shellcheck disable=SC2086
    pip_install -e "$CODE_DIR[env]" $idx
    local train_deps=""
    if [ "$DRY" = 1 ]; then train_deps="<[train] extras minus decord>"
    else
      train_deps="$("$PY" -c 'import sys,tomllib;d=tomllib.load(open(sys.argv[1],"rb"));print("\n".join(x for x in d["project"]["optional-dependencies"]["train"] if not x.lower().startswith("decord")))' "$CODE_DIR/pyproject.toml")"
    fi
    local old_ifs="$IFS"; IFS='
'
    # shellcheck disable=SC2086
    pip_install $train_deps
    IFS="$old_ifs"
    pip_install "decord2>=3.4" || pip_install "eva-decord>=0.6.1" \
      || note "warning: no decord replacement could be installed; DROID evaluation will not run on this Mac"
  else
    [ "$extras" = auto ] && extras="train,env"
    # shellcheck disable=SC2086
    pip_install -e "$CODE_DIR[$extras]" $idx
  fi
  pip_install -q "huggingface_hub[cli]>=0.34"
}

HF=""
find_hf() {
  if [ -x "$ENV_DIR/bin/hf" ]; then HF="$ENV_DIR/bin/hf"
  elif command -v hf >/dev/null 2>&1; then HF="$(command -v hf)"
  elif [ "$DRY" = 1 ]; then HF="$ENV_DIR/bin/hf"
  else die "the 'hf' CLI is missing: run without --skip-code, or pip install 'huggingface_hub[cli]'"; fi
}

download_ckpts() {
  find_hf
  local inc="" r e rest m s
  for r in $RUNS; do
    e="${r%%/*}"; rest="${r#*/}"; m="${rest%%/*}"; s="${rest#*/}"
    inc="$inc --include $e/${e}_$m/seed$s/*"
  done
  run mkdir -p "$CKPT_DIR"
  # shellcheck disable=SC2086
  (set -f; run "$HF" download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$CKPT_DIR" $inc)
}

download_tasks() {
  find_hf
  local e inc_sim="" droid=0
  for e in $SEL_ENVS; do
    if [ "$e" = droid ]; then droid=1; else inc_sim="$inc_sim --include $(task_path "$e")"; fi
  done
  if [ -n "$inc_sim" ]; then
    run mkdir -p "$TASK_DIR"
    # shellcheck disable=SC2086
    (set -f; run "$HF" download "$DATA_REPO" --repo-type dataset --revision "$DATA_REV" --local-dir "$TASK_DIR" $inc_sim --include eval_trajs/SHA256SUMS)
    printf '+ sha256 check %s\n' "$TASK_DIR/eval_trajs" >&2
    [ "$DRY" = 1 ] || sha_check "$TASK_DIR/eval_trajs" SHA256SUMS
  fi
  if [ "$droid" = 1 ]; then
    run mkdir -p "$HOME_DIR"
    run "$HF" download "$DATA_REPO" --repo-type dataset --revision "$DATA_REV" --local-dir "$HOME_DIR" \
      --include droid/droid_raw_eval16.tar --include droid/SHA256SUMS
    printf '+ sha256 check + untar %s\n' "$HOME_DIR/droid/droid_raw_eval16.tar" >&2
    [ "$DRY" = 1 ] || { sha_check "$HOME_DIR/droid" SHA256SUMS && (cd "$HOME_DIR/droid" && tar -xf droid_raw_eval16.tar); }
  fi
}

smoke() {
  local first; first="$(ckpt_file "${RUNS%% *}")"
  printf '+ smoke test: torch.load %s\n' "$first" >&2
  [ "$DRY" = 1 ] && return 0
  # The checkpoints are pickled modules (weights_only=False, as in the release README): this runs
  # code from the pinned, MIT-licensed release, and the code must be on the path.
  (cd "$CODE_DIR/h_jepa" && "$PY" - "$first" <<'PY'
import sys, torch
m = torch.load(sys.argv[1], weights_only=False, map_location="cpu")
n = sum(p.numel() for p in m.parameters()) if hasattr(m, "parameters") else 0
print(f"loaded {type(m).__name__}: {n/1e6:.1f}M parameters, torch {torch.__version__}")
PY
  )
}

# ---------- check ----------
check() {
  local ok=1 have=0 missing="" r f code_rev="" py_ok=0 tasks_ok=1 e
  if [ -d "$CODE_DIR/.git" ]; then code_rev="$(git -C "$CODE_DIR" rev-parse HEAD 2>/dev/null || true)"; fi
  [ "$code_rev" = "$CODE_REV" ] || ok=0
  if [ -x "$PY" ] && (cd "$CODE_DIR/h_jepa" 2>/dev/null && "$PY" -c 'import torch, hydra' >/dev/null 2>&1); then py_ok=1; else ok=0; fi
  for r in $RUNS; do
    f="$(ckpt_file "$r")"
    if [ -s "$f" ] && [ -s "$(dirname "$f")/normalizer.pt" ]; then have=$((have + 1)); else ok=0; missing="$missing $r"; fi
  done
  if [ "$SKIP_TASKS" = 0 ]; then
    for e in $SEL_ENVS; do
      case "$e" in
        droid) [ -d "$HOME_DIR/droid/droid_raw" ] || tasks_ok=0 ;;
        *) ls "$TASK_DIR/$(dirname "$(task_path "$e")")"/*.pt >/dev/null 2>&1 || tasks_ok=0 ;;
      esac
    done
    [ "$tasks_ok" = 1 ] || ok=0
  fi
  printf 'HJEPA_HOME   %s\n' "$HOME_DIR"
  printf 'code         %s %s\n' "$CODE_DIR" "$([ "$code_rev" = "$CODE_REV" ] && echo "ok (pinned ${CODE_REV:0:7})" || echo "MISSING or not at ${CODE_REV:0:7}")"
  printf 'python env   %s %s\n' "$ENV_DIR" "$([ "$py_ok" = 1 ] && echo ok || echo MISSING)"
  printf 'checkpoints  %s/%s present%s\n' "$have" "$(echo $RUNS | wc -w | tr -d ' ')" "${missing:+ (missing:$missing)}"
  [ "$SKIP_TASKS" = 1 ] || printf 'eval tasks   %s\n' "$([ "$tasks_ok" = 1 ] && echo ok || echo MISSING)"
  write_state "$([ "$ok" = 1 ] && echo installed || echo incomplete)"
  [ "$ok" = 1 ]
}

write_state() {
  [ -n "$STATE" ] && [ "$DRY" = 0 ] || return 0
  mkdir -p "$(dirname "$STATE")"
  local r runs="" sep=""
  for r in $RUNS; do runs="$runs$sep$(json_str "$r")"; sep=","; done
  {
    printf '{"status":%s,"checked":%s,' "$(json_str "$1")" "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
    printf '"home":%s,"codeDir":%s,"python":%s,"ckptDir":%s,' "$(json_str "$HOME_DIR")" "$(json_str "$CODE_DIR")" "$(json_str "$PY")" "$(json_str "$CKPT_DIR")"
    printf '"codeRev":%s,"modelRev":%s,"dataRev":%s,"runs":[%s],' "$(json_str "$CODE_REV")" "$(json_str "$MODEL_REV")" "$(json_str "$DATA_REV")" "$runs"
    printf '"firstCheckpoint":%s}\n' "$(json_str "$(ckpt_file "${RUNS%% *}")")"
  } > "$STATE"
}

# ---------- main ----------
if [ "$CHECK" = 1 ]; then check; exit $?; fi

note "selection: $RUNS"
note "download: ~${TOTAL_MB} MB into $HOME_DIR (the code and the Python environment come on top)"
[ "$SKIP_CODE" = 1 ] || install_code
download_ckpts
[ "$SKIP_TASKS" = 1 ] || download_tasks
[ "$SKIP_SMOKE" = 1 ] || smoke
if [ "$DRY" = 1 ]; then note "dry run: nothing was changed"; exit 0; fi
check
note "done. Use it with: export HJEPA_HOME=\"$HOME_DIR\"; cd \"$CODE_DIR/h_jepa\"; \"$PY\" eval.py --config-name <env>_<planner> seed=<seed> policy=<checkpoint>"
