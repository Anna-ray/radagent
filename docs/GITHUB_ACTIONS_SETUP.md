# RadAgent v2 — GitHub Actions Setup for Vultr Deployment

**Author:** Rayane Aggoune

This guide explains how to set up GitHub Actions for automatic deployment to Vultr when you push code.

---

## Prerequisites

1. Vultr instance provisioned and accessible via SSH
2. GitHub repository with admin access
3. SSH key pair for Vultr access

---

## Step 1: Generate SSH Key (if needed)

On your local machine:

```bash
# Generate new SSH key for GitHub Actions
ssh-keygen -t ed25519 -C "github-actions@radagent" -f ~/.ssh/radagent_deploy

# Copy public key to Vultr
ssh-copy-id -i ~/.ssh/radagent_deploy.pub root@<vultr-ip>

# Test connection
ssh -i ~/.ssh/radagent_deploy root@<vultr-ip> "echo 'Connection successful'"
```

---

## Step 2: Add GitHub Secrets

Go to your GitHub repository:
1. Click **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add the following secrets:

### Required Secrets

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `VULTR_HOST` | `<your-vultr-ip>` | Vultr instance IP address |
| `VULTR_USER` | `root` | SSH username (usually root) |
| `VULTR_SSH_KEY` | `<private-key-content>` | Contents of `~/.ssh/radagent_deploy` |
| `FEATHERLESS_API_KEY` | `<your-key>` | Featherless API key |
| `GEMINI_API_KEY` | `<your-key>` | Google Gemini API key |
| `SPEECHMATICS_API_KEY` | `<your-key>` | Speechmatics API key |

### How to Copy Private Key

```bash
# On Linux/Mac
cat ~/.ssh/radagent_deploy

# On Windows (PowerShell)
Get-Content ~\.ssh\radagent_deploy | clip

# Copy the entire output (including BEGIN and END lines)
```

---

## Step 3: Prepare Vultr Instance

SSH into your Vultr instance and prepare the deployment directory:

```bash
ssh root@<vultr-ip>

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Install Docker Compose
apt-get update
apt-get install -y docker-compose

# Create deployment directory
mkdir -p /opt/radagent
cd /opt/radagent

# Create .env file with API keys
cat > .env << 'EOF'
FEATHERLESS_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here
SPEECHMATICS_API_KEY=your-key-here
EOF

# Set permissions
chmod 600 .env
```

---

## Step 4: Test GitHub Actions Workflow

### Manual Trigger

1. Go to **Actions** tab in GitHub
2. Select **Deploy to Vultr** workflow
3. Click **Run workflow**
4. Select `feature/v2-milan` branch
5. Click **Run workflow**

### Automatic Trigger

Simply push to `feature/v2-milan` branch:

```bash
git add .
git commit -m "test: trigger deployment"
git push origin feature/v2-milan
```

---

## Step 5: Monitor Deployment

### View Workflow Logs

1. Go to **Actions** tab
2. Click on the running workflow
3. Click on **deploy** job
4. Expand steps to see logs

### Check Deployment on Vultr

```bash
# SSH into Vultr
ssh root@<vultr-ip>

# Check container status
cd /opt/radagent
docker-compose ps

# View logs
docker-compose logs -f radagent

# Test health endpoint
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "version": "2.0",
  "cached_demos": 5
}
```

---

## Workflow Details

### Trigger Events

- **Push to `feature/v2-milan`**: Automatic deployment
- **Manual dispatch**: Run workflow manually from Actions tab

### Deployment Steps

1. **Checkout code**: Clone repository
2. **Build Docker image**: Create container image
3. **Copy files to Vultr**: Transfer image + docker-compose.yml
4. **Deploy on Vultr**: 
   - Load Docker image
   - Stop old container
   - Start new container
   - Verify health check
5. **Notify status**: Report success/failure

### Deployment Time

- Build: ~2-3 minutes
- Transfer: ~1-2 minutes (depends on network)
- Deploy: ~30 seconds
- **Total: ~4-6 minutes**

---

## Troubleshooting

### Workflow Fails at "Copy files to Vultr"

**Problem:** SSH connection failed

**Solution:**
1. Verify `VULTR_HOST` secret is correct IP
2. Verify `VULTR_SSH_KEY` contains full private key (including headers)
3. Test SSH manually: `ssh -i ~/.ssh/radagent_deploy root@<vultr-ip>`

### Workflow Fails at "Deploy on Vultr"

**Problem:** Docker commands fail

**Solution:**
1. SSH into Vultr: `ssh root@<vultr-ip>`
2. Check Docker status: `systemctl status docker`
3. Check logs: `cd /opt/radagent && docker-compose logs`

### Health Check Fails

**Problem:** Container starts but health check fails

**Solution:**
1. Check if port 8000 is accessible: `curl http://localhost:8000/health`
2. Check firewall: `ufw status`
3. Check container logs: `docker-compose logs radagent`

### API Keys Not Working

**Problem:** Container starts but API calls fail

**Solution:**
1. Verify .env file on Vultr: `cat /opt/radagent/.env`
2. Restart container: `docker-compose restart`
3. Check logs for API errors: `docker-compose logs | grep -i error`

---

## Security Best Practices

1. **Never commit secrets to git**
   - Use GitHub Secrets for sensitive data
   - Add `.env` to `.gitignore`

2. **Rotate SSH keys regularly**
   - Generate new key every 90 days
   - Update `VULTR_SSH_KEY` secret

3. **Limit SSH access**
   - Use key-based auth only
   - Disable password auth: `PasswordAuthentication no` in `/etc/ssh/sshd_config`

4. **Monitor deployments**
   - Check Actions tab regularly
   - Set up email notifications for failed workflows

---

## Rollback Procedure

If deployment fails and you need to rollback:

```bash
# SSH into Vultr
ssh root@<vultr-ip>
cd /opt/radagent

# Stop current container
docker-compose down

# Load previous image (if saved)
docker load < radagent-v2-previous.tar.gz

# Start previous version
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

---

## Next Steps

Once GitHub Actions is working:

1. **Test full deployment**: Push a small change and verify it deploys
2. **Set up monitoring**: Add health check alerts
3. **Document public URL**: Update README with live demo URL
4. **Prepare for judging**: Ensure deployment is stable before May 19

---

## Support

If you encounter issues:

1. Check workflow logs in GitHub Actions
2. Check container logs on Vultr
3. Review this guide's troubleshooting section
4. Open GitHub issue if problem persists

**Contact:** rayane.aggoune@example.com