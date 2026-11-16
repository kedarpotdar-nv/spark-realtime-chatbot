#!/bin/bash
# Generate locally-trusted LAN HTTPS certificates using mkcert for spark-realtime-chatbot.
#
# Generates cert.pem and key.pem covering:
#   - localhost, 127.0.0.1, ::1
#   - <hostname> and <hostname>.local
#   - Primary LAN IPv4 address(es)
#   - Any extra hostnames/IPs passed as arguments

set -euo pipefail

SSL_CERT="${SSL_CERT:-cert.pem}"
SSL_KEY="${SSL_KEY:-key.pem}"

EXTRA_HOSTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ssl-cert)
            SSL_CERT="$2"
            shift 2
            ;;
        --ssl-key)
            SSL_KEY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--ssl-cert PATH] [--ssl-key PATH] [extra_host_or_ip ...]"
            echo ""
            echo "Generates locally-trusted TLS certificates via mkcert for:"
            echo "  localhost, 127.0.0.1, ::1, <hostname>, <hostname>.local, and detected LAN IPs."
            exit 0
            ;;
        *)
            EXTRA_HOSTS+=("$1")
            shift
            ;;
    esac
done

if ! command -v mkcert >/dev/null 2>&1; then
    echo "ERROR: 'mkcert' is not installed."
    echo ""
    echo "Install mkcert on DGX Spark (Ubuntu/Debian ARM64):"
    echo "  sudo apt update && sudo apt install -y libnss3-tools"
    echo "  curl -JLO \"https://dl.filippo.io/mkcert/latest?for=linux/arm64\""
    echo "  chmod +x mkcert-v*-linux-arm64"
    echo "  sudo mv mkcert-v*-linux-arm64 /usr/local/bin/mkcert"
    exit 1
fi

# Detect short hostname and .local mDNS name
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
HOST_LOCAL="${HOST_SHORT}.local"

# Detect LAN IPv4 addresses (excluding loopback, docker, and tailscale 100.x interfaces)
LAN_IPS=()
if command -v ip >/dev/null 2>&1; then
    while IFS= read -r ip_addr; do
        [ -n "$ip_addr" ] && LAN_IPS+=("$ip_addr")
    done < <(ip -4 -o addr show scope global 2>/dev/null | awk '!/docker|br-|veth|tailscale|virbr/ {split($4, a, "/"); print a[1]}')
fi

# Fallback to hostname -I if ip command filtered everything out
if [ ${#LAN_IPS[@]} -eq 0 ] && command -v hostname >/dev/null 2>&1; then
    for ip_addr in $(hostname -I 2>/dev/null || true); do
        case "$ip_addr" in
            127.*|172.17.*|100.*|*:*) ;;
            *) LAN_IPS+=("$ip_addr") ;;
        esac
    done
fi

SAN_LIST=("localhost" "127.0.0.1" "::1" "$HOST_SHORT" "$HOST_LOCAL")
for ip_addr in "${LAN_IPS[@]}"; do
    SAN_LIST+=("$ip_addr")
done
for extra in "${EXTRA_HOSTS[@]}"; do
    SAN_LIST+=("$extra")
done

echo "=========================================="
echo "Generating mkcert LAN Certificates"
echo "=========================================="
echo "Installing local CA in system trust store (if not already installed)..."
mkcert -install

echo ""
echo "Generating certificate for:"
for san in "${SAN_LIST[@]}"; do
    echo "  - $san"
done
echo ""

mkcert -cert-file "$SSL_CERT" -key-file "$SSL_KEY" "${SAN_LIST[@]}"

CAROOT_DIR="$(mkcert -CAROOT)"

echo ""
echo "✅ Certificates written to:"
echo "   Cert: $SSL_CERT"
echo "   Key : $SSL_KEY"
echo ""
echo "=========================================="
echo "Client Device Trust Setup (Mac & iPhone/iPad)"
echo "=========================================="
echo "Root CA location on this Spark:"
echo "  ${CAROOT_DIR}/rootCA.pem"
echo ""
echo "Copy rootCA.pem to your Mac/iPhone (do NOT share rootCA-key.pem):"
echo "  scp $(whoami)@${LAN_IPS[0]:-$HOST_LOCAL}:\"${CAROOT_DIR}/rootCA.pem\" ~/Downloads/spark-rootCA.pem"
echo ""
echo "1. macOS:"
echo "   sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain ~/Downloads/spark-rootCA.pem"
echo "   (Or double-click spark-rootCA.pem -> Keychain Access -> System -> Always Trust)"
echo ""
echo "2. iOS / iPadOS:"
echo "   a) AirDrop or email spark-rootCA.pem to your iPhone/iPad and tap to install the profile."
echo "   b) Go to Settings -> General -> VPN & Device Management -> install 'mkcert ...' profile."
echo "   c) Go to Settings -> General -> About -> Certificate Trust Settings ->"
echo "      toggle ON full trust for the mkcert root certificate."
echo ""
echo "Then start the server with:"
echo "  SSL_CERT=${SSL_CERT} SSL_KEY=${SSL_KEY} ./launch-https.sh"
