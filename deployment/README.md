# Jarvis Deployment

For the current Meta setup, use [META_SETUP.md](META_SETUP.md). The older EC2
instructions below describe a different deployment and are not its startup path.

This directory contains the minimal deployment scaffolding for running the
Jarvis backend on EC2 behind `jarvis.trygitbio.com`.

## Files

- `docker-compose.production.yml`: builds and runs the backend container.
- `env.production.example`: production environment template.
- `nginx.example.conf`: reverse proxy example with websocket headers.
- `ec2_minimal_setup.md`: EC2 setup notes.
- `deploy.sh`: build/restart and local health check.
- `restart.sh`: restart the backend container.
- `logs.sh`: tail backend logs.
- `check_production_health.sh`: verify health, DB readiness, buckets, and model config.
- `check_realtime_ws.sh`: create a disposable session and verify the public audio websocket.
- `production_operations.md`: deploy, restart, TLS, diagnostics, and recovery runbook.
- `backup_restore.md`: Postgres and storage backup/restore guidance.
- `rotate_secrets.md`: secret rotation checklist.

## First deploy

```bash
cd jarvis/backend
cp .env.example .env
```

Fill in every required production value, then:

```bash
cd ../deployment
./deploy.sh
```

The deployment script now fails if authenticated readiness is degraded. After
deployment, verify the public endpoint and websocket:

```bash
curl https://jarvis.trygitbio.com/health
./check_production_health.sh https://jarvis.trygitbio.com
./check_realtime_ws.sh https://jarvis.trygitbio.com
```
