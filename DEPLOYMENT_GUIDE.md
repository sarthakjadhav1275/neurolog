#  deployment Guide for NeuroLog

##  Quick Deployment Options

###  Option 1: Streamlit Cloud (Recommended for Beginners)
**Free tier available, easiest setup**

#### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select `app_stable.py` as main file
   - Click "Deploy"

#### URL Format: `https://yourusername-neurolog-app-stable-p5h2h4.streamlit.app`

---

###  Option 2: Railway (Easy & Affordable)
**$5/month starting, great for production**

#### Steps:
1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

2. **Deploy**
   ```bash
   railway init
   railway up
   ```

#### Configuration:
- Add `RAILWAY_ENVIRONMENT=production`
- Set port to `8501`

---

###  Option 3: Heroku (Free tier available)
**Classic choice for Python apps**

#### Steps:
1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   heroku login
   ```

2. **Create Procfile**
   ```bash
   echo "web: streamlit run app_stable.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
   ```

3. **Deploy**
   ```bash
   heroku create your-neurolog-app
   git push heroku main
   ```

---

###  Option 4: DigitalOcean (Professional)
**$6/month starting, full control**

#### Steps:
1. **Create Droplet**
   - Ubuntu 22.04
   - 1GB RAM minimum
   - $6/month

2. **Setup Server**
   ```bash
   # SSH into your droplet
   ssh root@your-droplet-ip
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install Docker Compose
   curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   chmod +x /usr/local/bin/docker-compose
   ```

3. **Deploy**
   ```bash
   # Clone your repo
   git clone https://github.com/yourusername/neurolog.git
   cd neurolog
   
   # Run with Docker Compose
   docker-compose up -d
   ```

---

###  Option 5: AWS (Enterprise)
**Scalable, pay-as-you-go**

#### Steps:
1. **Use AWS Elastic Beanstalk**
   - Go to AWS Console
   - Create new application
   - Choose "Python platform"
   - Upload your code

2. **Or use EC2 + Docker**
   ```bash
   # Launch EC2 instance
   # Install Docker
   # Run docker-compose
   ```

---

##  Production Checklist

### Security
- [ ] Use HTTPS (SSL/TLS)
- [ ] Set up firewall rules
- [ ] Use environment variables for secrets
- [ ] Enable authentication
- [ ] Regular backups

### Performance
- [ ] Use CDN for static assets
- [ ] Enable caching
- [ ] Monitor resource usage
- [ ] Set up logging
- [ ] Health checks

### Monitoring
- [ ] Set up uptime monitoring
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring
- [ ] User analytics

---

##  Environment Variables

### Required for Production
```bash
# Streamlit settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true

# Security
SECRET_KEY=your-secret-key-here
DATABASE_URL=your-database-url

# Optional
SENTRY_DSN=your-sentry-dsn
```

---

##  Domain Setup

### Custom Domain
1. **Buy domain** (Namecheap, GoDaddy, etc.)
2. **DNS Settings**
   ```
   A record: @ -> your-server-ip
   CNAME: www -> your-domain.com
   ```
3. **SSL Certificate**
   - Use Let's Encrypt (free)
   - Or Cloudflare (free)

---

##  Troubleshooting

### Common Issues
1. **Port binding errors**
   - Make sure port 8501 is open
   - Check firewall settings

2. **Memory issues**
   - Increase server RAM
   - Use streamlit caching

3. **Slow loading**
   - Optimize images
   - Use CDN
   - Enable compression

### Health Check
```bash
curl http://your-domain.com/_stcore/health
```

---

##  Cost Comparison

| Platform | Free Tier | Paid Plans | Best For |
|----------|-----------|------------|----------|
| Streamlit Cloud | Yes | $10/month | Beginners |
| Railway | $5/month | $20/month | Small projects |
| Heroku | Yes | $7/month | Hobby projects |
| DigitalOcean | No | $6/month | Production |
| AWS | $0/month | Pay-as-you-go | Enterprise |

---

##  Quick Start Command

```bash
# For immediate public access
streamlit run app_stable.py --server.port=8501 --server.address=0.0.0.0

# Then use ngrok for temporary public URL
ngrok http 8501
```

---

**Choose the option that best fits your needs and budget!**
