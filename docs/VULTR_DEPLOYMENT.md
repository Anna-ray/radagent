# RadAgent v2 — Vultr Deployment Guide

**Author:** Rayane Aggoune  
**Target:** Milan AI Week 2026 AI Agent Olympics  
**Instance:** Vultr 4 vCPU / 8 GB RAM (~$24/month)

---

## Overview

This guide covers deploying RadAgent v2 to a Vultr instance for the public demo URL that judges will access during evaluation.

**Architecture:**
- **Desktop (RTX 4070 Ti SUPER):** Heavy inference (specialist, federation training)
- **Vultr (CPU-only):** Serves cached demo results + light routing/autonomy/dictation

---

## Prerequisites

1. **Vultr Account** with sponsor credits activated
2. **Vultr Instance** provisioned:
   - OS: Ubuntu 22.04 LTS
   - Plan: 4 vCPU / 8 GB RAM
   - Storage: 80 GB SSD
   - Location: Choose closest to Milan (e.g., Frankfurt, Amsterdam)

3. **API Keys** (set as environment variables):
   ```bash
   export FEATHERLESS_API_KEY="your-key-here"
   export GEMINI_API_KEY="your-key-here"
   export SPEECHMATICS_API_KEY="your-key-here"
   ```

4. **SSH Access** to Vultr instance:
   ```bash
   ssh-copy-id root@<vultr-ip>
   ```

5. **Docker** installed on Vultr instance:
   ```bash
   ssh root@<vultr-ip>
   curl -fsSL https://get.docker.com | sh
   systemctl enable docker
   systemctl start docker
   ```

---

## Deployment Steps

### 1. Prepare Cached Demo Results

On your **desktop** (with GPU), generate cached results for 5 canonical images:

```bash
# Create demo images directory
mkdir -p data/demo_images

# Copy 5 representative chest X-rays to data/demo_images/
# - sample_001.jpg (Normal)
# - sample_002.jpg (Cardiomegaly)
# - sample_003.jpg (Pleural effusion)
# - sample_004.jpg (Pneumonia)
# - sample_005.jpg (Multiple findings)

# Run deployment script (generates cached results + deploys)
export VULTR_IP="<your-vultr-ip>"
chmod +x scripts/deploy_vultr.sh
./scripts/deploy_vultr.sh
```

The script will:
1. Generate cached results for all 5 images (~5-10 min on RTX 4070 Ti SUPER)
2. Build Docker image
3. Transfer to Vultr
4. Deploy with docker-compose
5. Optionally set up nginx reverse proxy

### 2. Verify Deployment

```bash
# Check container status
ssh root@$VULTR_IP 'cd /opt/radagent && docker-compose ps'

# View logs
ssh root@$VULTR_IP 'cd /opt/radagent && docker-compose logs -f'

# Test health endpoint
curl http://$VULTR_IP:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.0",
  "cached_demos": 5
}
```

### 3. Access Dashboard

Open in browser:
```
http://<vultr-ip>:8000
```

You should see:
- RadAgent v2.0 header
- "Grounded, Federated, Autonomous · Milan AI Week 2026" subtitle
- All 5 panels visible (modality badge, comparison, dictation, autonomy, federation)

### 4. Test Cached Demo

1. Drop one of the 5 canonical images (sample_001.jpg through sample_005.jpg)
2. Dashboard should load instantly from cache (~100ms response time)
3. All panels should populate with pre-computed results

---

## Optional: HTTPS with Let's Encrypt

If you have a domain name pointing to your Vultr IP:

```bash
ssh root@$VULTR_IP

# Install certbot
apt-get update
apt-get install -y certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
```

Update README with HTTPS URL: `https://your-domain.com`

---

## Monitoring

### View Logs
```bash
ssh root@$VULTR_IP 'cd /opt/radagent && docker-compose logs -f radagent'
```

### Check Resource Usage
```bash
ssh root@$VULTR_IP 'docker stats radagent-v2'
```

Expected usage:
- CPU: 5-15% (idle), 40-60% (during request)
- Memory: 1.5-2.5 GB
- Network: <1 MB/s

### Restart Container
```bash
ssh root@$VULTR_IP 'cd /opt/radagent && docker-compose restart'
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
ssh root@$VULTR_IP 'cd /opt/radagent && docker-compose logs'

# Common issues:
# 1. Missing API keys → Set in .env file
# 2. Port 8000 already in use → Change in docker-compose.yml
# 3. Out of memory → Upgrade to 16 GB RAM instance
```

### Cached Results Not Loading

```bash
# Verify cached results exist
ssh root@$VULTR_IP 'ls -lh /opt/radagent/runs/cached_demo/'

# Should show 5 directories (sample_001 through sample_005)
# Each with output.json, audit.json, gradcam/ folder
```

### Dashboard Not Accessible

```bash
# Check if container is running
ssh root@$VULTR_IP 'docker ps | grep radagent'

# Check firewall
ssh root@$VULTR_IP 'ufw status'

# If firewall is active, allow port 8000
ssh root@$VULTR_IP 'ufw allow 8000/tcp'
```

---

## Cost Estimate

**Vultr 4 vCPU / 8 GB RAM:**
- Hourly: $0.048
- Daily: $1.15
- Monthly: $35.00

**With Sponsor Credits:**
- 30-day free trial covers full hackathon period
- No out-of-pocket cost

---

## Updating Deployment

To update code or cached results:

```bash
# Pull latest code
git pull origin feature/v2-milan

# Re-run deployment script
export VULTR_IP="<your-vultr-ip>"
./scripts/deploy_vultr.sh
```

The script handles:
- Stopping old container
- Building new image
- Transferring updated files
- Starting new container
- Zero-downtime deployment (nginx keeps serving during update)

---

## Public URL for Submission

Once deployed, document the public URL in:

1. **README.md** (top of file):
   ```markdown
   🌐 **Live Demo:** http://<vultr-ip>:8000
   ```

2. **Submission Form:**
   - Project URL: `https://github.com/Anna-ray/radagent`
   - Live Demo URL: `http://<vultr-ip>:8000`
   - Video URL: `<canva-pro-video-url>`

3. **DEMO_SCRIPT.md:**
   - Scene 6 (Close): Show live URL on screen

---

## Security Notes

- **API Keys:** Never commit to git. Use `.env` file (gitignored).
- **SSH:** Use key-based auth, disable password auth.
- **Firewall:** Only expose port 8000 (or 80/443 if using nginx).
- **Updates:** Keep Docker and Ubuntu packages updated.

---

## Cleanup (After Hackathon)

```bash
# Stop and remove container
ssh root@$VULTR_IP 'cd /opt/radagent && docker-compose down'

# Remove deployment directory
ssh root@$VULTR_IP 'rm -rf /opt/radagent'

# Destroy Vultr instance (via Vultr dashboard)
```

---

## Support

If deployment fails, check:
1. Logs: `docker-compose logs`
2. Health endpoint: `curl http://localhost:8000/health`
3. GitHub Issues: https://github.com/Anna-ray/radagent/issues

For urgent issues during judging, contact: rayane.aggoune@example.com