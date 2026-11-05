# EC2 Minimal Setup for Jarvis

## Prerequisites

- An EC2 instance with inbound `80` and `443` open.
- DNS record: `jarvis.trygitbio.com` points to the EC2 public IP.
- Docker Engine installed.
- Docker Compose plugin installed.
- Nginx installed.
- Certbot or an existing TLS setup.

## Install Docker on Ubuntu

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg nginx
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and back in so the Docker group change applies.

## Configure environment

```bash
cd jarvis/backend
cp .env.example .env
```

Use `jarvis/deployment/env.production.example` as the deployment checklist.
Phase 1 can run with provider placeholders still set to `replace_me`.

## Start backend

```bash
cd jarvis/deployment
./deploy.sh
```

Local EC2 health check:

```bash
curl http://127.0.0.1:9000/health
```

## Configure Nginx

Copy `nginx.example.conf` into an Nginx site file and enable it:

```bash
sudo cp nginx.example.conf /etc/nginx/sites-available/jarvis.trygitbio.com
sudo ln -s /etc/nginx/sites-available/jarvis.trygitbio.com /etc/nginx/sites-enabled/jarvis.trygitbio.com
sudo nginx -t
sudo systemctl reload nginx
```

Then add TLS:

```bash
sudo certbot --nginx -d jarvis.trygitbio.com
```

Public health check:

```bash
curl https://jarvis.trygitbio.com/health
```
