#!/bin/bash
# RadAgent v2 — Vultr Deployment Script
# Author: Rayane Aggoune
#
# This script:
# 1. Generates pre-cached demo results for 5 canonical images
# 2. Builds Docker image
# 3. Deploys to Vultr instance
# 4. Sets up nginx reverse proxy (optional)
# 5. Outputs public URL

set -e  # Exit on error

echo "============================================"
echo "RadAgent v2 — Vultr Deployment"
echo "============================================"
echo ""

# Configuration
VULTR_IP="${VULTR_IP:-}"
VULTR_USER="${VULTR_USER:-root}"
DEMO_IMAGES_DIR="data/demo_images"
CACHED_RESULTS_DIR="runs/cached_demo"

# Check required environment variables
if [ -z "$VULTR_IP" ]; then
    echo "ERROR: VULTR_IP environment variable not set"
    echo "Usage: VULTR_IP=<your-vultr-ip> ./scripts/deploy_vultr.sh"
    exit 1
fi

if [ -z "$FEATHERLESS_API_KEY" ]; then
    echo "WARNING: FEATHERLESS_API_KEY not set. Vanilla baseline will fail."
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "WARNING: GEMINI_API_KEY not set. Autonomy planner will fail."
fi

if [ -z "$SPEECHMATICS_API_KEY" ]; then
    echo "WARNING: SPEECHMATICS_API_KEY not set. Dictation auditor will fail."
fi

echo "Step 1: Preparing cached demo results..."
echo "----------------------------------------"

# Create cached results directory
mkdir -p "$CACHED_RESULTS_DIR"

# List of 5 canonical demo images (these should exist in your data directory)
DEMO_IMAGES=(
    "sample_001.jpg"  # Normal chest X-ray
    "sample_002.jpg"  # Cardiomegaly
    "sample_003.jpg"  # Pleural effusion
    "sample_004.jpg"  # Pneumonia
    "sample_005.jpg"  # Multiple findings
)

echo "Generating cached results for ${#DEMO_IMAGES[@]} demo images..."

for img in "${DEMO_IMAGES[@]}"; do
    img_path="$DEMO_IMAGES_DIR/$img"
    
    if [ ! -f "$img_path" ]; then
        echo "WARNING: Demo image not found: $img_path"
        echo "Skipping..."
        continue
    fi
    
    img_id="${img%.*}"  # Remove extension
    output_dir="$CACHED_RESULTS_DIR/$img_id"
    
    echo "  Processing $img..."
    
    # Run v1 pipeline to generate cached result
    python scripts/predict_one.py \
        --image "$img_path" \
        --output "$output_dir" \
        --checkpoint runs/nih14_convnextv2_base_384/best_model.pt \
        --config configs/nih14_convnextv2_base.yaml \
        --calibration runs/nih14_convnextv2_base_384/calibration.json \
        --rag-index data/rag_index \
        --language en \
        > /dev/null 2>&1 || echo "    WARNING: Failed to generate cached result for $img"
    
    # Also generate vanilla baseline if API key is available
    if [ -n "$FEATHERLESS_API_KEY" ]; then
        python scripts/run_vanilla_baseline.py \
            --image "$img_path" \
            --output "$CACHED_RESULTS_DIR/vanilla_baseline" \
            > /dev/null 2>&1 || echo "    WARNING: Failed to generate vanilla baseline for $img"
    fi
    
    echo "    ✓ Cached result generated"
done

echo ""
echo "Step 2: Building Docker image..."
echo "----------------------------------------"

docker build -t radagent-v2:latest .

echo ""
echo "Step 3: Saving Docker image for transfer..."
echo "----------------------------------------"

docker save radagent-v2:latest | gzip > radagent-v2.tar.gz

echo ""
echo "Step 4: Transferring to Vultr instance..."
echo "----------------------------------------"

echo "Copying files to $VULTR_USER@$VULTR_IP..."

# Create deployment directory on Vultr
ssh "$VULTR_USER@$VULTR_IP" "mkdir -p /opt/radagent"

# Transfer files
scp radagent-v2.tar.gz "$VULTR_USER@$VULTR_IP:/opt/radagent/"
scp docker-compose.yml "$VULTR_USER@$VULTR_IP:/opt/radagent/"
scp -r "$CACHED_RESULTS_DIR" "$VULTR_USER@$VULTR_IP:/opt/radagent/runs/"

# Transfer .env file if it exists
if [ -f ".env" ]; then
    scp .env "$VULTR_USER@$VULTR_IP:/opt/radagent/"
fi

echo ""
echo "Step 5: Deploying on Vultr..."
echo "----------------------------------------"

ssh "$VULTR_USER@$VULTR_IP" << 'ENDSSH'
cd /opt/radagent

# Load Docker image
echo "Loading Docker image..."
docker load < radagent-v2.tar.gz

# Stop existing container if running
echo "Stopping existing container..."
docker-compose down || true

# Start new container
echo "Starting RadAgent v2..."
docker-compose up -d

# Wait for health check
echo "Waiting for health check..."
sleep 10

# Check if container is running
if docker ps | grep -q radagent-v2; then
    echo "✓ RadAgent v2 is running"
else
    echo "✗ RadAgent v2 failed to start"
    docker-compose logs
    exit 1
fi

# Clean up
rm radagent-v2.tar.gz

echo ""
echo "Deployment complete!"
ENDSSH

echo ""
echo "Step 6: Setting up nginx (optional)..."
echo "----------------------------------------"

read -p "Set up nginx reverse proxy with SSL? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh "$VULTR_USER@$VULTR_IP" << 'ENDSSH'
    # Install nginx and certbot
    apt-get update
    apt-get install -y nginx certbot python3-certbot-nginx
    
    # Create nginx config
    cat > /etc/nginx/sites-available/radagent << 'EOF'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/radagent /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test and reload nginx
    nginx -t && systemctl reload nginx
    
    echo "✓ Nginx configured"
    echo ""
    echo "To enable HTTPS, run:"
    echo "  certbot --nginx -d your-domain.com"
ENDSSH
fi

echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Public URL: http://$VULTR_IP:8000"
echo ""
echo "To view logs:"
echo "  ssh $VULTR_USER@$VULTR_IP 'cd /opt/radagent && docker-compose logs -f'"
echo ""
echo "To stop:"
echo "  ssh $VULTR_USER@$VULTR_IP 'cd /opt/radagent && docker-compose down'"
echo ""
echo "To update:"
echo "  Run this script again with the same VULTR_IP"
echo ""

# Made with Bob
