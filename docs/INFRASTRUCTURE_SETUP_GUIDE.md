# Cloud-Native Infrastructure Setup Guide

## Overview

This guide will walk you through setting up all external infrastructure needed to deploy the FIAP Challenge application to Azure with full CI/CD capabilities. **No local software installations required!** Everything runs in the cloud using GitHub Actions and Azure services.

## 🎯 What This Guide Does

✅ **Creates Azure infrastructure** (VMs, storage, functions, networking)  
✅ **Configures CI/CD pipeline** (automated deployments)  
✅ **Sets up monitoring and security**  
✅ **Optimizes costs** (runs on free tier)  

## 🚀 What You Actually Need

- **Azure account** (you have this)
- **GitHub repository** (you have this)  
- **Web browser** (for Azure Portal and GitHub)
- **~30 minutes** of your time

## Step-by-Step Setup Process

### Phase 1: Azure Account Configuration

#### 1.1 Login to Azure Portal

1. Go to [Azure Portal](https://portal.azure.com)
2. Login with your Azure account
3. Verify you have an active subscription

#### 1.2 Create Service Principal for GitHub Actions

**Option A: Using Azure Portal (Recommended)**
1. Go to **Azure Active Directory** → **App registrations**
2. Click **New registration**
3. Name: `github-actions-fiap`
4. Click **Register**
5. Note the **Application (client) ID** and **Directory (tenant) ID**
6. Go to **Certificates & secrets** → **New client secret**
7. Add description: `GitHub Actions`
8. Click **Add**
8. **Copy the secret value immediately** (you won't see it again)

**Option B: Using Azure Cloud Shell**
1. Go to [Azure Cloud Shell](https://shell.azure.com)
2. Run these commands:
```bash
# Create service principal
az ad sp create-for-rbac --name "github-actions-fiap" --role="Contributor" --scopes="/subscriptions/YOUR_SUBSCRIPTION_ID" --sdk-auth

# Save the output - you'll need it for GitHub secrets
```

#### 1.3 Get Azure Subscription Information

**In Azure Portal:**
1. Go to **Subscriptions**
2. Note your **Subscription ID**
3. Go to **Azure Active Directory** → **Overview**
4. Note your **Tenant ID**

#### 1.4 Generate SSH Key (for VM access)

**Option A: Using Azure Cloud Shell**
```bash
# Generate SSH key
ssh-keygen -t rsa -b 4096 -f ~/.ssh/fiap_azure_rsa

# Display public key
cat ~/.ssh/fiap_azure_rsa.pub
```

**Option B: Using Windows (if you have Git installed)**
```cmd
ssh-keygen -t rsa -b 4096 -f %USERPROFILE%\.ssh\fiap_azure_rsa
type %USERPROFILE%\.ssh\fiap_azure_rsa.pub
```

**Save the public key content** - you'll need it for GitHub secrets.

### Phase 2: Local Development Setup

#### 2.1 Prerequisites (You Already Have These)

✅ **Docker Desktop** - You mentioned you already have this  
✅ **Git** - For version control  
✅ **Code Editor** - VS Code, PyCharm, etc.  
✅ **Python 3.8+** - For running the application locally  

#### 2.2 Clone and Setup Repository

```bash
# Clone your repository
git clone https://github.com/vnzoliveira/STDR.DataScience.PJlook.git
cd STDR.DataScience.PJlook

# Install Python dependencies
pip install -r requirements.txt
```

#### 2.3 Test Application Locally

**Option A: Run with Docker (Recommended)**
```bash
# Build and run the application
docker-compose up

# Access dashboard at: http://localhost:8050
# Access Nomad UI at: http://localhost:4646 (if running)
```

**Option B: Run Python Directly**
```bash
# Run the ETL pipeline
python src/cli/main.py build-all

# Run the dashboard
python src/ui/app.py
```

#### 2.4 Test Azure Integration Locally

**Get Azure Connection String:**
1. Go to Azure Portal → Storage Account
2. Go to **Access keys** → **key1** → **Connection string**
3. Copy the connection string

**Set Environment Variables:**
```bash
# Windows (PowerShell)
$env:AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."

# Windows (Command Prompt)
set AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# Linux/Mac
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
```

**Test Azure Blob Integration:**
```bash
# Test downloading from Azure
python -c "from src.cli.azure_sync import download_processed_data; download_processed_data()"

# Test uploading to Azure
python -c "from src.cli.azure_sync import upload_model_outputs; upload_model_outputs()"
```

#### 2.5 Local Development Workflow

**Typical Development Cycle:**
```bash
# 1. Make code changes
# Edit your Python files

# 2. Test locally
python src/cli/main.py build-all

# 3. Test with Docker
docker-compose up

# 4. Commit and push
git add .
git commit -m "Add new feature"
git push origin main

# 5. GitHub Actions automatically deploys to Azure
```

**Debugging Tips:**
- Check logs in `reports/logs/` directory
- Use `docker-compose logs` to see container logs
- Set `AZURE_STORAGE_CONNECTION_STRING` to test Azure integration
- Use `python -m pdb src/cli/main.py build-all` for debugging

### Phase 3: GitHub Repository Configuration

#### 3.1 Configure GitHub Secrets

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

**Add Azure Authentication Secrets:**
- **Name:** `AZURE_CREDENTIALS`
- **Value:** Full JSON from service principal creation (Option B in Phase 1.2)

- **Name:** `ARM_CLIENT_ID`
- **Value:** Application (client) ID from Azure Portal

- **Name:** `ARM_CLIENT_SECRET`
- **Value:** Client secret value from Azure Portal

- **Name:** `ARM_SUBSCRIPTION_ID`
- **Value:** Your Azure subscription ID

- **Name:** `ARM_TENANT_ID`
- **Value:** Your Azure tenant ID

**Add SSH Access Secrets:**
- **Name:** `SSH_PUBLIC_KEY`
- **Value:** Content of your SSH public key from Phase 1.4

**Note:** Additional secrets will be added automatically after deployment.

#### 3.2 Enable GitHub Actions

1. Go to **Settings** → **Actions** → **General**
2. Ensure Actions are enabled
3. Set workflow permissions to "Read and write permissions"

#### 3.3 Configure Branch Protection (Optional)

1. Go to **Settings** → **Branches**
2. Click **Add rule**
3. Branch name pattern: `main`
4. Enable "Require status checks to pass before merging"

### Phase 4: Deploy Infrastructure via GitHub Actions

#### 4.1 Trigger Deployment

**Option A: Push to Main Branch (Recommended)**
1. Go to your GitHub repository
2. Make a small change (e.g., edit README.md)
3. Commit and push to main branch
4. GitHub Actions will automatically start the deployment

**Option B: Manual Workflow Trigger**
1. Go to **Actions** tab in your repository
2. Select **Terraform Apply** workflow
3. Click **Run workflow**
4. Choose environment: `dev`
5. Choose action: `apply`
6. Click **Run workflow**

#### 4.2 Monitor Deployment

**Watch GitHub Actions:**
1. Go to **Actions** tab
2. Click on the running workflow
3. Monitor each job:
   - ✅ **Terraform Check** - Validates configuration
   - ✅ **Terraform Apply** - Creates Azure resources
   - ✅ **Build Images** - Builds Docker containers
   - ✅ **Deploy Function** - Deploys Azure Function
   - ✅ **Deploy Nomad** - Deploys applications

**Expected Timeline:**
- Terraform deployment: ~15-20 minutes
- Total deployment: ~30-40 minutes

#### 4.3 Verify Deployment

**Check Azure Portal:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to resource group: `fiapchallenge-dev`
3. Verify all resources are created:
   - ✅ Virtual Machines (1 for dev)
   - ✅ Storage Account with 3 containers
   - ✅ Azure Function App
   - ✅ Key Vault
   - ✅ Load Balancer

**Check GitHub Actions Output:**
- Look for "Terraform Output" section
- Note the dashboard URL and VM IPs

### Phase 5: Test Your Deployment

#### 5.1 Access Your Dashboard

**Get Dashboard URL:**
1. Go to GitHub Actions → Latest workflow run
2. Look for "Terraform Output" section
3. Copy the `dashboard_url` value
4. Open in your browser

**Expected Result:**
- Dashboard loads successfully
- You can navigate between "Desafio 1" and "Desafio 2" pages
- Charts and visualizations display

#### 5.2 Test Data Processing

**Upload Test Data:**
1. Go to Azure Portal → Storage Account
2. Navigate to `raw-data` container
3. Upload an Excel file (or use the existing test data)
4. The Azure Function will automatically process it

**Verify Processing:**
1. Check `processed-data` container for parquet files
2. Check `model-outputs` container for ML results
3. Refresh your dashboard to see new data

#### 5.3 Monitor System Health

**Check Nomad UI:**
1. Get VM IP from GitHub Actions output
2. Open `http://VM_IP:4646` in browser
3. Verify both jobs are running:
   - `ml-pipeline` (batch job)
   - `dashboard` (service)

**Check Azure Function:**
1. Go to Azure Portal → Function App
2. Check function execution logs
3. Verify no errors in processing

### Phase 6: Configure Monitoring (Optional)

#### 6.1 Set Up Cost Alerts

**In Azure Portal:**
1. Go to **Cost Management + Billing**
2. Click **Budgets**
3. Create budget: $50/month for dev environment
4. Set alerts at 50%, 80%, 100% of budget

#### 6.2 Enable Auto-Shutdown

**Verify Auto-Shutdown:**
- Dev VMs automatically shut down at 7 PM BRT
- This saves ~60% on compute costs
- VMs restart when you access them

#### 6.3 Monitor Application Logs

**Azure Function Logs:**
1. Go to Function App → Monitoring → Logs
2. Set up alerts for errors

**Nomad Job Logs:**
1. Access Nomad UI at `http://VM_IP:4646`
2. Click on jobs to view logs
3. Monitor for any failures

## 🎉 You're Done! 

**Congratulations!** Your FIAP Challenge application is now running in the cloud with:

✅ **Automated ETL** - Azure Functions process Excel files  
✅ **ML Pipeline** - Runs daily with ensemble models  
✅ **Interactive Dashboard** - Accessible via web browser  
✅ **CI/CD Pipeline** - Automatic deployments on code changes  
✅ **Cost Optimized** - Runs on Azure free tier  

## 📊 What You Have Now

### **Infrastructure:**
- 1 Virtual Machine (Nomad cluster)
- Azure Function App (serverless ETL)
- Blob Storage (3 containers for data)
- Load Balancer (public access)
- Key Vault (secrets management)

### **Applications:**
- **ML Pipeline**: Runs daily at 2 AM, processes data, generates models
- **Dashboard**: 24/7 web application with interactive charts
- **Azure Function**: Automatically converts Excel → Parquet

### **Automation:**
- **GitHub Actions**: Builds, tests, and deploys on every push
- **Auto-shutdown**: VMs shut down at 7 PM to save costs
- **Health monitoring**: Built-in health checks and logging

## 💰 Cost Breakdown

### **Monthly Costs (Dev Environment):**
- VM (B2ms): ~$30-35
- Storage: ~$2-3
- Functions: ~$0-1 (free tier)
- Load Balancer: ~$20-25
- **Total: ~$55-65/month**

### **With Auto-Shutdown:**
- **Actual cost: ~$30-40/month** (60% savings)

### **Free Trial:**
- **$200 credit lasts 5-6 months** for dev environment

## 🔄 Ongoing Operations

### **Daily Operations:**
- Check dashboard for new data
- Monitor GitHub Actions for deployments
- Review Azure cost alerts

### **Weekly Operations:**
- Review application logs
- Check for any errors or issues
- Monitor resource usage

### **Monthly Operations:**
- Review costs and optimize
- Update dependencies if needed
- Plan scaling for production

## 🚀 Next Steps

### **Immediate (Optional):**
1. **Upload real data** and test the full pipeline
2. **Customize dashboard** with your branding
3. **Set up monitoring alerts** for errors

### **Short Term (1-3 months):**
1. **Scale to production** (3 VMs for high availability)
2. **Add HTTPS/SSL** certificates
3. **Implement user authentication**

### **Long Term (3-6 months):**
1. **Multi-region deployment** for global access
2. **Advanced monitoring** with Grafana
3. **API endpoints** for external integrations

## 🆘 Getting Help

### **If Something Goes Wrong:**
1. **Check GitHub Actions** logs for deployment issues
2. **Check Azure Portal** for resource status
3. **Check Nomad UI** for application status
4. **Review this guide** for troubleshooting

### **Support Resources:**
- **GitHub Issues**: [Report bugs](https://github.com/vnzoliveira/STDR.DataScience.PJlook/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/vnzoliveira/STDR.DataScience.PJlook/discussions)
- **Azure Support**: [Azure Support Center](https://azure.microsoft.com/support/)

## 📚 Documentation

- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **GitHub Secrets**: [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md)

---

**🎊 Your cloud-native data science platform is ready for production!**

**Total setup time**: ~30 minutes  
**Monthly cost**: ~$30-40 (with auto-shutdown)  
**Free trial duration**: 5-6 months with $200 credit

---

## ✅ Verification Checklist

After completing the setup, verify:

- [ ] **Azure resources created** (check Azure Portal)
- [ ] **GitHub Actions workflow completed** successfully
- [ ] **Dashboard accessible** via web browser
- [ ] **Nomad UI accessible** at VM IP:4646
- [ ] **Azure Function deployed** and listed in portal
- [ ] **Storage containers created** (raw-data, processed-data, model-outputs)
- [ ] **Cost alerts configured** (optional)
- [ ] **Auto-shutdown enabled** for dev VMs

## 🐛 Troubleshooting

### **GitHub Actions Fails:**
1. Check all secrets are configured correctly
2. Verify Azure service principal has proper permissions
3. Review workflow logs for specific error messages

### **Dashboard Not Accessible:**
1. Check VM is running in Azure Portal
2. Verify load balancer health probe is passing
3. Check Nomad job status in Nomad UI

### **Function Not Processing Files:**
1. Check Function App logs in Azure Portal
2. Verify storage account connection string
3. Test with manual file upload

### **High Costs:**
1. Verify auto-shutdown is enabled
2. Check for unused resources
3. Review cost breakdown in Azure Portal

## 📞 Support

- **GitHub Issues**: [Report bugs](https://github.com/vnzoliveira/STDR.DataScience.PJlook/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/vnzoliveira/STDR.DataScience.PJlook/discussions)
- **Azure Support**: [Azure Support Center](https://azure.microsoft.com/support/)

---

**🎉 Setup Complete! Your cloud-native data science platform is ready!**

