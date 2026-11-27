#!/usr/bin/env bash
set -euo pipefail
command -v google-earth-pro >/dev/null
command -v supervisorctl >/dev/null
command -v Xvfb >/dev/null
command -v nvidia-smi >/dev/null
supervisorctl pid >/dev/null
if ! test -x /opt/VirtualGL/bin/vglrun; then
  bash "$(dirname "${BASH_SOURCE[0]}")/install_virtualgl.sh"
fi
mkdir -p /root/pairUAV_img/earth_runtime/.config /root/pairUAV_img/earth_runtime/.cache

cat > /opt/supervisor-scripts/earth-xvfb.sh <<'EOF'
#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
# A container stop/start leaves /tmp/.X1-lock behind with no X server; Xvfb
# then refuses to start. Remove the lock only when its owner is not an Xvfb.
if [ -e /tmp/.X1-lock ]; then
  owner="$(tr -d ' ' < /tmp/.X1-lock 2>/dev/null || true)"
  if [ -z "$owner" ] || [ "$(cat "/proc/${owner}/comm" 2>/dev/null)" != "Xvfb" ]; then
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
  fi
fi
exec /usr/bin/Xvfb :1 -screen 0 3840x2160x24 -ac -noreset +extension GLX -nolisten tcp
EOF
cat > /opt/supervisor-scripts/earth-window-manager.sh <<'EOF'
#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export DISPLAY=:1
for attempt in $(seq 1 45); do
  /usr/bin/xdpyinfo >/dev/null 2>&1 && break
  sleep 1
done
/usr/bin/xdpyinfo >/dev/null
exec /usr/bin/openbox --sm-disable
EOF
cat > /opt/supervisor-scripts/earth-pro.sh <<'EOF'
#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export DISPLAY=:1
for attempt in $(seq 1 45); do
  /usr/bin/wmctrl -m 2>/dev/null | /usr/bin/rg -q 'Name: Openbox' && break
  sleep 1
done
/usr/bin/wmctrl -m | /usr/bin/rg -q 'Name: Openbox'
exec /usr/bin/env -u LIBGL_ALWAYS_SOFTWARE HOME=/root/pairUAV_img/earth_runtime XDG_CONFIG_HOME=/root/pairUAV_img/earth_runtime/.config QT_X11_NO_MITSHM=1 LC_ALL=C.UTF-8 QT_ACCESSIBILITY=1 /usr/bin/prlimit --core=0:0 -- /opt/VirtualGL/bin/vglrun -d egl /usr/bin/google-earth-pro
EOF
chmod +x /opt/supervisor-scripts/earth-{xvfb,window-manager,pro}.sh
for service in earth-xvfb earth-window-manager earth-pro; do
  case "$service" in
    earth-xvfb) priority=100 ;;
    earth-window-manager) priority=110 ;;
    earth-pro) priority=120 ;;
  esac
  cat > "/etc/supervisor/conf.d/${service}.conf" <<EOF
[program:${service}]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/${service}.sh
priority=${priority}
autostart=true
autorestart=unexpected
startsecs=3
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
EOF
done
supervisorctl reread
supervisorctl update
for service in earth-xvfb earth-window-manager earth-pro; do
  if ! supervisorctl status "$service" | rg -q 'RUNNING|STARTING'; then
    supervisorctl start "$service"
  fi
done
for attempt in $(seq 1 45); do
  DISPLAY=:1 xdpyinfo >/dev/null 2>&1 && break
  sleep 1
done
DISPLAY=:1 xdpyinfo >/dev/null
DISPLAY=:1 env -u LIBGL_ALWAYS_SOFTWARE /opt/VirtualGL/bin/vglrun -d egl glxinfo -B
for attempt in $(seq 1 60); do
  DISPLAY=:1 wmctrl -l | rg -q 'Google Earth Pro$' && break
  if DISPLAY=:1 wmctrl -l | rg -q 'Google Earth Crash Detected$'; then
    /usr/bin/python3 "$(dirname "${BASH_SOURCE[0]}")/recover_google_earth.py"
    break
  fi
  sleep 1
done
DISPLAY=:1 wmctrl -l | rg 'Google Earth Pro$'
