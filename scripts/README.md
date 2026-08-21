# H3 Studio scripts

Operational helpers belong here. Root-level shell files remain stable entrypoints
for existing deployments; all paths and timing values should come from
`config.yaml` or explicit environment variables.

## Director heartbeat

The monitor does not start a paused project. It polls the project every five
minutes, writes an hourly checkpoint, and exits after all shots are completed:

```bash
python scripts/director_heartbeat.py --project-id <project-id>
```

For unattended generation, explicitly enable review approval:

```bash
python scripts/director_heartbeat.py --auto-approve --auto-retry
```

The monitor exits after all shots complete. Use tmux/systemd/supervisord when a
long-lived background process is required.
