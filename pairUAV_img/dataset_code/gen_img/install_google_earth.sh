#!/usr/bin/env bash
set -euo pipefail

if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then
  echo "Google Earth Pro's official Debian package currently requires amd64." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
# Preserve the host-provided supervisor configuration in unattended installs.
APT_OPTS=(-y -o Dpkg::Options::=--force-confold -o Dpkg::Options::=--force-confdef)
apt-get install "${APT_OPTS[@]}" \
  ca-certificates curl xdg-utils xvfb xdotool wmctrl scrot \
  xserver-xorg-core openbox python3-pil tesseract-ocr ripgrep x11-utils mesa-utils supervisor \
  fonts-liberation libgl1 libglu1-mesa libx11-xcb1 libxcb1 \
  libxcb-keysyms1 \
  libxcb-render0 libxcb-shape0 libxcb-xfixes0 libxcb-randr0 \
  libxcb-shm0 libxcb-xkb1 libxkbcommon-x11-0 libasound2t64

package="$(mktemp --suffix=.deb)"
trap 'rm -f "$package"' EXIT
curl -fL --retry 3 \
  https://dl.google.com/dl/earth/client/current/google-earth-pro-stable_current_amd64.deb \
  -o "$package"
apt-get install "${APT_OPTS[@]}" "$package"

dpkg-query -W -f='Google Earth Pro ${Version}\n' google-earth-pro-stable
