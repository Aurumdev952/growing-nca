#!/usr/bin/env bash
# End-to-end check on a private Xvfb display, so the user's desktop never receives input.
# usage: drive_xvfb.sh <shot-dir> [models-dir]
set -euo pipefail
OUT=$1
MODELS=${2:-$(dirname "$0")/../../models}
BIN=$(dirname "$0")/../target/release/nca-viewer
DISP=:99
mkdir -p "$OUT"

Xvfb $DISP -screen 0 1280x900x24 -nolisten tcp &
XVFB=$!
trap 'kill $APP $XVFB 2>/dev/null || true' EXIT
export DISPLAY=$DISP
timeout 10 bash -c 'until xdpyinfo >/dev/null 2>&1; do sleep 0.2; done'

XMODIFIERS= LIBGL_ALWAYS_SOFTWARE=1 "$BIN" --models "$MODELS" >"$OUT/app.log" 2>&1 &
APP=$!
W=$(timeout 20 bash -c "until xdotool search --pid $APP --name 'Growing NCA' 2>/dev/null | head -1 | grep .; do sleep 0.3; done")
xdotool windowfocus --sync "$W"

shot() { ffmpeg -loglevel error -y -f x11grab -video_size 1280x900 -i "$DISP" -frames:v 1 "$OUT/$1.png"; echo "shot $1"; }
wait_s() { timeout "$1" bash -c 'while :; do sleep 0.5; done' || true; }
key() { xdotool key "$1"; wait_s 0.3; }

eval "$(xdotool getwindowgeometry --shell "$W")"
CX=$((X + 310 + (WIDTH - 310) / 2))
CY=$((Y + HEIGHT / 2))

shot 0a-start
wait_s 3
shot 0b-start-plus-3s
key r
wait_s 10
key space
shot 1-grown-paused

xdotool mousemove $((CX - 120)) $((CY - 160)) mousedown 1
for i in $(seq 0 24); do xdotool mousemove $((CX - 120 + i * 10)) $((CY - 160 + i * 8)); done
xdotool mouseup 1
wait_s 0.5
shot 2-erased-paused

key space
wait_s 15
shot 3-regrown

xdotool mousemove $((X + 186)) $((Y + 55)) click 1
wait_s 12
shot 4-switched-to-fish
echo "app log:"; cat "$OUT/app.log"
