#!/bin/bash
# Launch script for Webcam → VLM (Qwen3-VL) with HTTPS
# Enables browser webcam access from remote machines

# Default configuration - override with environment variables
VLM_SERVER="${VLM_SERVER:-http://localhost:8080}"
PORT="${PORT:-5000}"

# SSL certificate paths
SSL_KEY="${SSL_KEY:-key.pem}"
SSL_CERT="${SSL_CERT:-cert.pem}"

# Check if certificates exist, generate if not
if [ ! -f "$SSL_KEY" ] || [ ! -f "$SSL_CERT" ]; then
    echo "=========================================="
    echo "SSL certificates not found!"
    echo "=========================================="
    echo "Generating self-signed certificate..."
    echo ""
    
    # Get hostname for certificate CN
    HOSTNAME=$(hostname)
    
    openssl req -x509 -newkey rsa:2048 \
        -keyout "$SSL_KEY" -out "$SSL_CERT" \
        -days 365 -nodes \
        -subj "/CN=$HOSTNAME" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate certificates. Install openssl or provide certificates manually."
        echo ""
        echo "To generate manually:"
        echo "  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes"
        exit 1
    fi
    echo "✅ Certificates generated: $SSL_KEY, $SSL_CERT"
    echo ""
fi

echo "=========================================="
echo "Webcam → Qwen3-VL (HTTPS)"
echo "=========================================="
echo "VLM Server:  $VLM_SERVER"
echo "Web Port:    $PORT (HTTPS)"
echo "SSL Key:     $SSL_KEY"
echo "SSL Cert:    $SSL_CERT"
echo "=========================================="
echo ""
echo "📷 This uses YOUR BROWSER's webcam"
echo "   (not a camera attached to the server)"
echo ""
echo "🌐 Open in browser:"
echo "   https://$(hostname):$PORT"
echo ""
echo "⚠️  Accept the self-signed certificate warning"
echo "   Then click 'Start Camera' and allow access"
echo ""
echo "=========================================="

# Launch the web server
python3 webcam_vlm_web.py \
    --port "$PORT" \
    --vlm-server "$VLM_SERVER" \
    --https

