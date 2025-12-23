# HTTPS Setup for Remote Access

## Why HTTPS is Required

Modern browsers **require HTTPS** (or `localhost`) to access the microphone. For remote access to Spark, you need HTTPS.

## Solutions

### Option 1: SSH Port Forwarding (Easiest - No HTTPS Needed)

Access via `localhost` through SSH tunnel - browsers allow microphone on `localhost`:

**From your local machine:**
```bash
ssh -L 8001:localhost:8001 user@spark-hostname
```

**Then access:**
```
http://localhost:8001
```

✅ **Works immediately** - no certificate setup needed  
✅ **Secure** - encrypted through SSH tunnel  
✅ **Microphone works** - browser treats it as localhost

---

### Option 2: Uvicorn with SSL Certificates

Run uvicorn directly with SSL:

```bash
uvicorn server:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
```

**Generate self-signed certificate (for testing):**
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

⚠️ **Note:** Self-signed certs will show browser warnings - you'll need to accept them.

---

### Option 3: Reverse Proxy (nginx) - Recommended for Production

Use nginx as reverse proxy with SSL:

**nginx config (`/etc/nginx/sites-available/ces-voice`):**
```nginx
server {
    listen 443 ssl;
    server_name your-spark-hostname;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Then run your app on port 8001:**
```bash
uvicorn server:app --host 127.0.0.1 --port 8001
```

---

### Option 4: Use Spark's Built-in HTTPS/Proxy

If Spark provides HTTPS URLs or a reverse proxy, use those. Check Spark documentation for:
- Port forwarding URLs
- HTTPS endpoints
- Reverse proxy configuration

---

## Quick Test Setup (Self-Signed Cert)

**1. Generate certificate:**
```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=your-spark-hostname"
```

**2. Update launch script to use SSL:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem
```

**3. Access:**
```
https://your-spark-hostname:8443
```

**4. Accept browser warning** (click "Advanced" → "Proceed to site")

---

## Recommended Approach

**For development/testing:** Use **Option 1 (SSH port forwarding)** - simplest and works immediately.

**For production:** Use **Option 3 (nginx reverse proxy)** with proper SSL certificates (Let's Encrypt, etc.).

