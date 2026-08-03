# Deployment

Deployed on **AWS EC2 Free Tier** (`t2.micro`, Ubuntu 22.04) running Streamlit as a `systemd` service, kept alive independent of any SSH session.

## Infrastructure

- **Instance:** t2.micro (1 vCPU, 1GB RAM) — AWS Free Tier
- **Storage:** 20GB gp2 (within the 30GB free-tier allowance)
- **Security group:** SSH (22) restricted to a known IP; TCP 8501 open for the Streamlit app
- **Swap:** 4GB swap file added, since the app loads two transformer models (DistilBART ~306M params, RoBERTa ~125M params) simultaneously, which exceeds the 1GB RAM available on a `t2.micro` without it

## Known dependency fixes

Two issues surfaced during setup that a fresh clone would otherwise hit immediately on a clean instance:

1. **`nltk`'s VADER lexicon isn't bundled** — `SentimentIntensityAnalyzer()` throws a `LookupError` without it. Fixed by calling `nltk.download("vader_lexicon", quiet=True)` at app startup (cached after first run).
2. **`newspaper3k` breaks on `lxml>=5.2`** — that version moved `lxml.html.clean` into a separate package newspaper3k doesn't declare as a dependency. Fixed by pinning `lxml<5.2` in `requirements.txt`.

## Setup steps

```bash
# System packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git

# Swap (critical for running two transformer models in 1GB RAM)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# App setup
git clone https://github.com/MaDHuPRi/Automated_News_Analyzer.git
cd Automated_News_Analyzer
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Test manually first
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Running as a service

To keep the app running independent of any SSH session, it runs as a `systemd` service:

```ini
# /etc/systemd/system/news-analyzer.service
[Unit]
Description=News Analyzer Streamlit App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Automated_News_Analyzer
ExecStart=/home/ubuntu/Automated_News_Analyzer/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable news-analyzer
sudo systemctl start news-analyzer
sudo systemctl status news-analyzer   # confirm: active (running)
```

## Notes on the public IP

The current live URL uses the instance's dynamically assigned public IP, which changes if the instance is stopped and restarted (not on a simple reboot). An Elastic IP can be attached at no extra cost while the instance stays running, to keep the link stable long-term.
