#!/usr/bin/env bash
# Create test utterances with macOS `say` (16 kHz mono WAV), plus "phone" copies that
# are band-limited to 300-3400 Hz and mixed with pink noise, so the engines disagree.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p samples

say_wav() {  # <name> <voice> <text>
  say -v "$2" -o "samples/$1.wav" --data-format=LEI16@16000 "$3"
}

say_wav email   Rishi    "My email is chethan dot rao at gmail dot com, and my number is nine eight four five, two one three, three seven zero."
say_wav booking Samantha "Book a table for four at Toit in Indiranagar this Saturday at seven thirty p.m."
say_wav tech    Daniel   "Can LiveKit race Deepgram Nova three against Groq Whisper and G P T four o transcribe?"
say_wav payment Rishi    "Please transfer twelve thousand five hundred rupees to Shyam Sundar, account ending four four seven one."

for name in email booking tech payment; do
  ffmpeg -loglevel error -y -i "samples/$name.wav" -filter_complex \
    "[0:a]highpass=f=300,lowpass=f=3400,aresample=8000,aresample=16000[speech];\
anoisesrc=color=pink:amplitude=0.05:sample_rate=16000[noise];\
[speech][noise]amix=inputs=2:duration=first:normalize=0" \
    -ar 16000 -ac 1 -sample_fmt s16 "samples/${name}_phone.wav"
done

ls -1 samples
