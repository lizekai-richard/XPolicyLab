#!/usr/bin/env bash
# Event stream for the run official_seed0_mot_canvas_nsc_s36k_xpl_cfg6_nfe10_fs8 (one line per event; used by a Claude Monitor).
# Emits: every unit outcome (DONE / RE-QUEUE / FAILED / STALE), slot trouble (quarantine, draining, unreachable, SIGKILL,
# dispatch FAILED, WARN slot), operator/autogrow controls (add_slots growth), the final ALL UNITS SETTLED, and a line when
# the master or the autogrow loop process disappears (a crashed master writes nothing to master.log). Exits when the master
# is gone.
set -u
RUN=${RUN:-official_seed0_mot_canvas_nsc_s36k_xpl_cfg6_nfe10_fs8}
S=$HOME/robodojo_master/$RUN
PAT='DONE |FAILED|RE-QUEUE|STALE|quarantin|draining|dispatch FAILED|ALL UNITS SETTLED|all slots idle|stop requested|control: |WARN slot|unreachable|SIGKILL'
watch_pid(){ # watch_pid PIDFILE NAME: print one line when that process is gone
  local f=$1 name=$2
  while [ -s "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; do sleep 60; done
  echo "$(date -u +%FT%TZ) EVENT: the $name process is gone ($f); last master.stdout.log line: $(tail -n 1 "$S/master.stdout.log" 2>/dev/null | cut -c1-160)"
}
( tail -n 0 -F "$S/master.log" 2>/dev/null | grep -E --line-buffered "$PAT" ) &
TAIL=$!
watch_pid "$S/autogrow.pid" autogrow &
AG=$!
watch_pid "$S/master.pid" master
sleep 5
pkill -P "$TAIL" 2>/dev/null; kill "$TAIL" "$AG" 2>/dev/null
exit 0
