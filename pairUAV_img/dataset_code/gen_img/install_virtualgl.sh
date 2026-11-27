#!/usr/bin/env bash
set -euo pipefail
version=3.1.5
package=$(mktemp --suffix=.deb)
trap 'rm -f "$package"' EXIT
curl -fL "https://github.com/VirtualGL/virtualgl/releases/download/${version}/virtualgl_${version}_amd64.deb" -o "$package"
printf '%s  %s\n' df3f7788ce41b182a47c0d298e5cd6d2d63579522cb41825970b7726e825485e "$package" | sha256sum -c -
dpkg -i "$package"
if DISPLAY=:1 xdpyinfo >/dev/null 2>&1; then
  DISPLAY=:1 env -u LIBGL_ALWAYS_SOFTWARE /opt/VirtualGL/bin/vglrun -d egl glxinfo -B
fi
