# AWS Setup Guide - Airbnb Price Prediction Project

## Overview
This project uses three AWS services:
- **S3**: Store the dataset
- **EC2**: Run MLflow tracking server
- **IAM**: Manage access credentials

---

## Understanding AWS Credentials (CRITICAL)

### Two Types of Credentials - Don't Confuse Them!

#### 1. IAM Access Keys (for programmatic access)
**What they are:**
- Access Key ID + Secret Access Key
- Like username/password for your code to talk to AWS

**What they're used for:**
- Your Python code accessing S3 (boto3)
- AWS CLI commands
- Any programmatic interaction with AWS services
Markdown Preview Enhanced
**How to create them:**
1. Go to AWS Console → IAM → Users
2. Select your user → Security credentials tab
3. Click "Create access key"
4. Choose "Command Line Interface (CLI)"
5. Copy both: Access Key ID and Secret Access Key (shown only once!)

**How to configure:**
```bash
aws configure
```
Enter:
- AWS Access Key ID: [paste yours]
- AWS Secret Access Key: [paste yours]
- Default region: us-east-1 (or your preference)
- Default output format: json

**Where they're stored:**
- Location: `~/.aws/credentials` (in your HOME directory, NOT the project)
- This file is automatically read by boto3 and AWS CLI

**NEVER:**
- Put these in your project folder
- Upload them to GitHub
- Share them in screenshots
- Hardcode them in Python scripts

---

#### 2. EC2 Key Pair (.pem file) (for SSH access)
**What it is:**
- A private key file (like a physical key)
- Specific to ONE EC2 instance (or can be reused across instances)

**What it's used for:**
- SSH connection to EC2 instance
- Opening the "door" to your virtual machine

**How to create:**
1. When launching EC2 instance → Key pair section
2. Click "Create new key pair"
3. Name it (e.g., "mlflow-airbnb-key")
4. Type: RSA
5. Format: .pem (for Mac/Linux) or .ppk (for Windows/PuTTY)
6. Download and SAVE the .pem file

**Where to store it:**
```bash
# Move to .ssh folder and set permissions
mv ~/Downloads/mlflow-airbnb-key.pem ~/.ssh/
chmod 400 ~/.ssh/mlflow-airbnb-key.pem
```

**How to use it:**
```bash
ssh -i ~/.ssh/mlflow-airbnb-key.pem ubuntu@YOUR_EC2_IP
```

**NEVER:**
- Lose this file (can't be re-downloaded)
- Upload to GitHub
- Share with anyone

---

## Step-by-Step Setup

### STEP 1: Configure IAM Access Keys (Do this ONCE)

If you haven't already:
1. Create Access Keys in IAM Console
2. Run `aws configure` and enter your keys
3. Test: `aws s3 ls` (should list buckets or show empty)

### STEP 2: Create S3 Bucket

**Via AWS CLI:**
```bash
# Create bucket (bucket names must be globally unique)
aws s3 mb s3://your-unique-bucket-name-airbnb-2024

# Verify it was created
aws s3 ls

# Upload dataset
aws s3 cp AB_NYC_2019.csv s3://your-unique-bucket-name-airbnb-2024/raw_data/listings.csv

# Verify upload
aws s3 ls s3://your-unique-bucket-name-airbnb-2024/raw_data/
```

**Via AWS Console:**
1. Go to S3 → Create bucket
2. Name it (must be globally unique)
3. Region: us-east-1
4. Keep defaults → Create
5. Upload file manually

**Via Python (boto3):**
```python
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')
bucket_name = 'your-bucket-name'

# Create bucket
s3.create_bucket(Bucket=bucket_name)

# Upload file
s3.upload_file(
    'path/to/AB_NYC_2019.csv',
    bucket_name,
    'raw_data/listings.csv'
)

# Verify
response = s3.list_objects_v2(Bucket=bucket_name, Prefix='raw_data/')
for obj in response['Contents']:
    print(f"{obj['Key']} - {obj['Size']} bytes")
```

### STEP 3: Launch EC2 Instance for MLflow

**Configuration:**
- Name: mlflow-server-airbnb
- AMI: Ubuntu Server 24.04 LTS
- Instance type: t2.micro (free tier)
- Key pair: Create new or use existing
- Storage: 8GB (default)

**Security Group (Firewall) Settings:**
Add these inbound rules:
| Type | Protocol | Port | Source | Purpose |
|------|----------|------|--------|---------|
| SSH | TCP | 22 | Your IP | For you to connect |
| Custom TCP | TCP | 5000 | 0.0.0.0/0 | MLflow UI access |

**After launching:**
- Note down the Public IPv4 address
- Save the .pem file to ~/.ssh/

### STEP 4: Connect to EC2 and Install MLflow

**Connect via SSH:**
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

**Install dependencies:**
```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip -y

# Install MLflow
pip3 install mlflow boto3

# Verify installation
mlflow --version
```

**Start MLflow Server:**
```bash
# Basic command (may give "Invalid Host header" error)
python3 -m mlflow server --host 0.0.0.0 --port 5000

# If you get "Invalid Host header" error, use:
python3 -m mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "*"
```

**Common Issues:**
- "Command 'mlflow' not found" → Use `python3 -m mlflow` instead
- "Address already in use" → Kill process: `sudo pkill -9 python3` then retry
- "Invalid Host header" → Add `--allowed-hosts "*"` flag

**Keep it running in background:**
```bash
sudo apt install screen -y
screen -S mlflow
python3 -m mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "*"
# Press Ctrl+A then D to detach
```

**Access MLflow UI:**
Open browser: `http://YOUR_EC2_PUBLIC_IP:5000`

### STEP 5: Configure Local Python to Use Remote MLflow

In your notebooks/scripts:
```python
import mlflow

# Set tracking URI to your EC2 instance
mlflow.set_tracking_uri("http://YOUR_EC2_PUBLIC_IP:5000")

# Now all mlflow.log_* commands will send to EC2
```

---

## Security Best Practices

1. **Never commit credentials:**
   - IAM keys stay in ~/.aws/
   - .pem files stay in ~/.ssh/
   - Both locations are outside your project folder

2. **Use .gitignore properly:**
   - Already configured in project .gitignore
   - Double-check before every commit

3. **Limit Security Group access:**
   - For production: restrict SSH to your IP only
   - For learning: 0.0.0.0/0 is okay for MLflow port

4. **Stop EC2 when not using:**
   - Free tier is 750 hours/month
   - Stop instance to save hours (not terminate!)

---

## Troubleshooting

**Can't connect to EC2:**
- Check Security Group allows your IP on port 22
- Verify .pem file permissions: `chmod 400 key.pem`
- Confirm using correct IP address

**MLflow UI not accessible:**
- Check Security Group allows port 5000
- Verify MLflow is running: `ps aux | grep mlflow`
- Check firewall: `sudo ufw status`

**S3 access denied:**
- Verify AWS credentials: `aws sts get-caller-identity`
- Check IAM permissions include S3 read/write

**boto3 can't find credentials:**
- Run `aws configure` again
- Check file exists: `ls ~/.aws/credentials`

---

## Cost Management

**Free Tier Limits:**
- EC2: 750 hours/month t2.micro
- S3: 5GB storage, 20,000 GET requests, 2,000 PUT requests
- Data transfer: 15GB out per month

**This project should cost $0 if:**
- You use t2.micro
- Stop EC2 when not working
- Delete S3 bucket after project submission

**To stop EC2:**
- AWS Console → EC2 → Instances → Select → Instance state → Stop
- (Don't terminate - that deletes everything!)

---

## Quick Reference Commands

```bash
# Test AWS credentials
aws sts get-caller-identity

# List S3 buckets
aws s3 ls

# Connect to EC2
ssh -i ~/.ssh/key.pem ubuntu@EC2_IP

# Check MLflow is running on EC2
ps aux | grep mlflow

# Stop MLflow on EC2
pkill -f mlflow
```

---

## Summary: What You Configured

**S3:**
- Bucket: `software-tools-ai`
- Dataset: `s3://software-tools-ai/raw_data/listings.csv`
- Access: boto3 uses credentials from ~/.aws/credentials

**EC2:**
- Instance IP: 35.183.177.64
- Key pair: airbnb.pem (stored in ~/.ssh/)
- Security groups: ports 22 (SSH) and 5000 (MLflow) open

**MLflow:**
- Running on: http://35.183.177.64:5000
- Command: `python3 -m mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "*"`
- Session: screen session named "mlflow"

**Local Python:**
```python
import mlflow
mlflow.set_tracking_uri("http://35.183.177.64:5000")
# Now all experiments log to EC2
```
