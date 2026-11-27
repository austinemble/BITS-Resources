# Audit automation (10-minute loop)

## Commands

```bash
# Mirror Inbox slides/transcripts → git-tracked portal_mirror (run after Drive sync)
python3 scripts/mirror_portal_assets.py

# Verify all fixable audit points; writes AUDIT_OPEN_POINTS.md
python3 scripts/audit_verify.py
echo $?   # 0 = pass, 2 = path, 3 = slides not in git, 4 = content
```

## Cursor loop (local IDE)

Run every **10 minutes** until `AUDIT_OPEN_POINTS.md` reports pass:

```bash
while true; do
  sleep 600
  echo 'AGENT_LOOP_TICK_audit-fix {"prompt":"Run python3 scripts/mirror_portal_assets.py and python3 scripts/audit_verify.py in BITS/Resources. Fix any FAIL in AUDIT_OPEN_POINTS.md; commit when clean."}'
done
```

First pass: run `mirror_portal_assets.py` + `audit_verify.py` once immediately, then arm the loop above.

## Open items intentionally not auto-fixed

- **Stats R1** (Miller & Freund): optional reference — add PDF to `Books/` when you have it.
- **Lecture videos** (`Videos/`, `Inbox/*.mp4`): stay local/Drive; hub shows transcripts + slide PDFs on Cloud after mirror.