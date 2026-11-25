#!/bin/sh
# Usage: sh src/render.sh   (run from the output/ folder)
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
here="$(cd "$(dirname "$0")" && pwd -W)"
"$EDGE" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$here/../pdf/Assignment2_Submission_v6_EN.pdf" "file:///$here/Assignment2_EN.html"
[ -f "$here/Assignment2_ZH.html" ] && "$EDGE" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$here/../pdf/Assignment2_Submission_v6_中文.pdf" "file:///$here/Assignment2_ZH.html"
exit 0
