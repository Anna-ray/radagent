# Vultr Deployment Guide — Step-by-Step

**RadAgent v2 — Milan AI Week 2026**  
**Author:** Rayane Aggoune

This guide walks you through deploying RadAgent v2 to Vultr from scratch.

---

## Prerequisites

- [ ] Vultr account with sponsor credits activated
- [ ] SSH client (Windows: PowerShell, Linux/Mac: Terminal)
- [ ] API keys ready:
  - Featherless API key
  - Google AI Studio API key
  - (Optional) Speechmatics API key

---

## Step 1: Generate SSH Key (If You Don't Have One)

**On Windows PowerShell:**
```powershell
# Check if you already have an SSH key
Test-Path ~\.ssh\id_rsa.pub

# If False, generate a new key
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# Press Enter 3 times (default location, no passphrase)

# Display your public key
Get-Content ~\.ssh\id_rsa.pub
# Copy this entire output
```

**On Linux/Mac:**
```bash
# Check if you already have an SSH key
ls ~/.ssh/id_rsa.pub

# If not found, generate a new key
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# Press Enter 3 times (default location, no passphrase)

# Display your public key
cat ~/.ssh/id_rsa.pub
# Copy this entire output
```

---

## Step 2: Provision Vultr Instance

1. **Go to Vultr Dashboard:**
   - https://my.vultr.com/

2. **Click "Deploy +" → "Deploy New Server"**

3. **Choose Server Type:**
   - Select: **Cloud Compute - Regular Performance**

4. **Choose Location:**
   - Recommended: **Frankfurt, Germany** (closest to Sétif, Algeria)
   - Alternative: **Paris, France** or **Amsterdam, Netherlands**

5. **Choose Image:**
   - Select: **Ubuntu 22.04 LTS x64**

6. **Choose Plan:**
   - Select: **4 vCPU, 16 GB RAM, 320 GB SSD**
   - Cost: $96/month (covered by sponsor credits)
   - **DO NOT choose smaller plans** — dashboard needs 16 GB RAM

7. **Add SSH Key:**
   - Click "Add New" under SSH Keys
   - Paste your public key from Step 1
   - Name it: "radagent-deploy-key"
   - Click "Add SSH Key"

8. **Server Hostname:**
   - Enter: `radagent-v2-demo`

9. **Enable Auto Backups:** (Optional)
   - Skip for demo (saves credits)

10. **Deploy Now:**
    - Click "Deploy Now"
    - Wait 2-3 minutes for provisioning

11. **Copy IP Address:**
    - Once status shows "Running", copy the IP address
    - Example: `149.28.123.456`

---

## Step 3: Test SSH Connection

**On Windows PowerShell:**
```powershell
# Set your Vultr IP
$VULTR_IP = "149.28.123.456"  # Replace with your actual IP

# Test connection
ssh root@$VULTR_IP "echo 'Connection successful!'"

# If you see "Connection successful!", proceed to Step 4
# If you see "Permission denied", check your SSH key setup
```

**On Linux/Mac:**
```bash
# Set your Vultr IP
export VULTR_IP="149.28.123.456"  # Replace with your actual IP

# Test connection
ssh root@$VULTR_IP "echo 'Connection successful!'"

# If you see "Connection successful!", proceed to Step 4
```

**Troubleshooting:**
- If connection fails: Check firewall allows SSH (port 22)
- If "Permission denied": Verify SSH key was added correctly in Vultr dashboard
- If "Connection refused": Wait 1-2 more minutes for server to fully boot

---

## Step 4: Prepare Environment Variables

Create a `.env` file in your local project root:

```bash
# Required API keys
FEATHERLESS_API_KEY=your_featherless_key_here
GOOGLE_API_KEY=your_google_key_here

# Optional (for Scene 2.5 dictation)
SPEECHMATICS_API_KEY=your_speechmatics_key_here

# VLM configuration (using Featherless)
VLLM_URL=https://api.featherless.ai/v1
VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

**Where to get API keys:**

1. **Featherless:**
   - Go to https://featherless.ai/
   - Sign up / Log in
   - Go to "API Keys" section
   - Create new key
   - Copy key (starts with `sk-...`)

2. **Google AI Studio:**
   - Go to https://aistudio.google.com/app/apikey
   - Click "Create API Key"
   - Copy key (starts with `AIza...`)

3. **Speechmatics (Optional):**
   - Go to https://portal.speechmatics.com/
   - Sign up with hackathon email
   - Activate $200 credits
   - Go to "API Keys"
   - Create new key
   - Copy key

---

## Step 5: Run Deployment Script

**On Windows PowerShell:**
```powershell
# Navigate to project root
cd C:\Users\pc\.bob\radagent

# Set Vultr IP
$env:VULTR_IP = "149.28.123.456"  # Replace with your IP

# Run deployment (this will take 5-10 minutes)
bash scripts/deploy_vultr.sh
```

**On Linux/Mac:**
```bash
# Navigate to project root
cd ~/radagent

# Set Vultr IP
export VULTR_IP="149.28.123.456"  # Replace with your IP

# Run deployment (this will take 5-10 minutes)
bash scripts/deploy_vultr.sh
```

**What the script does:**
1. Installs Docker on Vultr
2. Copies project files to Vultr
3. Copies `.env` file with API keys
4. Builds Docker image (~3 minutes)
5. Starts dashboard container
6. Prints public URL

**Expected output:**
```
✓ Docker installed on Vultr
✓ Project files copied to Vultr
✓ Environment variables copied
✓ Docker image built successfully
✓ Dashboard container started

🎉 DEPLOYMENT COMPLETE!

Public URL: http://149.28.123.456:8080
Dashboard: http://149.28.123.456:8080
Health check: http://149.28.123.456:8080/health

Test with:
  curl http://149.28.123.456:8080/health
```

---

## Step 6: Verify Deployment

**Test health endpoint:**
```bash
curl http://149.28.123.456:8080/health
# Expected: {"status":"ok"}
```

**Open dashboard in browser:**
```
http://149.28.123.456:8080
```

You should see the RadAgent v2 dashboard with:
- Upload area for chest X-rays
- Language selector (English/Arabic/Bilingual)
- "Compare with ungrounded baseline" checkbox
- 5 v2 panels ready to display results

**Test with a sample image:**
1. Drag-drop a chest X-ray into the upload area
2. Wait for WebSocket progress updates
3. See findings with citations appear
4. Click citations to see evidence cards
5. View Grad-CAM heatmaps

---

## Step 7: Update README with Public URL

```bash
# Edit README.md locally
# Find the line: "Live demo: [Coming soon — Vultr URL will be added here]"
# Replace with: "Live demo: http://149.28.123.456:8080"

git add README.md
git commit -m "docs: Add Vultr public URL"
git push origin feature/v2-milan
```

---

## Step 8: Monitor Deployment

**SSH into Vultr to check logs:**
```bash
ssh root@$VULTR_IP

# Check container status
docker ps

# View dashboard logs
docker logs radagent-dashboard -f

# Exit logs: Ctrl+C
# Exit SSH: exit
```

**Check resource usage:**
```bash
ssh root@$VULTR_IP "docker stats --no-stream"
```

Expected:
- CPU: 5-15% (idle), 50-80% (during inference)
- Memory: 2-4 GB (idle), 8-12 GB (during inference)

---

## Troubleshooting

### Issue: "Connection refused" when accessing dashboard

**Solution:**
```bash
# Check if container is running
ssh root@$VULTR_IP "docker ps"

# If not running, check logs
ssh root@$VULTR_IP "docker logs radagent-dashboard"

# Restart container
ssh root@$VULTR_IP "docker restart radagent-dashboard"
```

### Issue: "API key invalid" errors in logs

**Solution:**
```bash
# Check environment variables
ssh root@$VULTR_IP "docker exec radagent-dashboard env | grep API"

# If keys are missing, re-copy .env file
scp .env root@$VULTR_IP:/root/radagent/.env

# Restart container
ssh root@$VULTR_IP "docker restart radagent-dashboard"
```

### Issue: Dashboard loads but inference fails

**Solution:**
```bash
# Check Featherless API connectivity
ssh root@$VULTR_IP "docker exec radagent-dashboard curl -H 'Authorization: Bearer YOUR_KEY' https://api.featherless.ai/v1/models"

# If fails, check API key is correct in .env
```

### Issue: Out of memory errors

**Solution:**
```bash
# Check memory usage
ssh root@$VULTR_IP "free -h"

# If memory is full, restart container
ssh root@$VULTR_IP "docker restart radagent-dashboard"

# If problem persists, upgrade to 32 GB RAM plan
```

---

## Maintenance

### Update deployment after code changes:

```bash
# On local machine
git add .
git commit -m "fix: Update dashboard"
git push origin feature/v2-milan

# Re-run deployment
export VULTR_IP="149.28.123.456"
bash scripts/deploy_vultr.sh
```

### Stop dashboard:

```bash
ssh root@$VULTR_IP "docker stop radagent-dashboard"
```

### Start dashboard:

```bash
ssh root@$VULTR_IP "docker start radagent-dashboard"
```

### View logs:

```bash
ssh root@$VULTR_IP "docker logs radagent-dashboard -f"
```

### Destroy instance (after hackathon):

1. Go to Vultr dashboard
2. Click on your server
3. Click "Settings" → "Destroy Server"
4. Confirm destruction

---

## Security Notes

**For production deployment (not needed for hackathon demo):**

1. **Enable HTTPS:**
   ```bash
   # Install nginx + Let's Encrypt
   ssh root@$VULTR_IP "apt-get install -y nginx certbot python3-certbot-nginx"
   
   # Configure domain (requires DNS setup)
   ssh root@$VULTR_IP "certbot --nginx -d radagent.yourdomain.com"
   ```

2. **Enable firewall:**
   ```bash
   ssh root@$VULTR_IP "ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw enable"
   ```

3. **Disable root SSH:**
   ```bash
   # Create non-root user first
   ssh root@$VULTR_IP "adduser radagent && usermod -aG sudo radagent"
   
   # Then disable root login
   ssh root@$VULTR_IP "sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config && systemctl restart sshd"
   ```

**For hackathon demo:** HTTP on port 8080 is fine. Judges don't care about HTTPS.

---

## Cost Estimate

**Vultr 4 vCPU, 16 GB RAM instance:**
- Hourly: $0.143
- Daily: $3.43
- 7 days (until submission): $24.01
- 30 days (full month): $96.00

**With $30 Vultr credits:** You can run for ~8 days free.

**Recommendation:** Deploy 2-3 days before submission, destroy after judging.

---

## Submission Form

**Live Demo URL:**
```
http://149.28.123.456:8080
```

**Note to include:**
```
Live demo deployed on Vultr CPU instance (4 vCPU, 16 GB RAM).
Serves cached results for instant playback during judging.
Heavy GPU inference (specialist training, federation) runs on 
local RTX 4070 Ti SUPER and is documented in the video demo.
```

---

**Deployment complete! Your RadAgent v2 dashboard is now live at the public URL. 🚀**