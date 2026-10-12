#!/usr/bin/env bash
# Align the original 250 FPS RGB endpoint frames (9,19,...) with 25 FPS event windows.
set -euo pipefail
if (( $# != 3 )); then
    echo "Usage: $0 ORIGINAL_250FPS_RGB.mp4 EVENT_25FPS.mp4 OUTPUT.mp4" >&2
    exit 2
fi
rgb_source=$1
event_video=$2
output=$3
mkdir -p "$(dirname "$output")"
temp="${output%.mp4}.part.mp4"
trap 'rm -f "$temp"' EXIT
ffmpeg -nostdin -y -hide_banner -loglevel error \
    -i "$rgb_source" -i "$event_video" \
    -filter_complex "[0:v]select='eq(mod(n,10),9)',setpts=N/(25*TB),drawtext=text='Original RGB - frame 10j+9':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[rgb];[1:v]setpts=PTS-STARTPTS,drawtext=text='Event - 40 ms window':x=10:y=10:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[event];[rgb][event]hstack=inputs=2[v]" \
    -map '[v]' -an -c:v libx264 -preset fast -crf 17 -pix_fmt yuv420p -r 25 -shortest "$temp"
mv "$temp" "$output"
trap - EXIT
printf '%s\n' "$output"
