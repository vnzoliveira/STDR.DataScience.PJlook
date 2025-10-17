# Deployment Guide - FIAP Challenge Azure Infrastructure

Complete step-by-step guide to deploy the FIAP Challenge application to Azure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Deploy Infrastructure](#deploy-infrastructure)
4. [Deploy Azure Function](#deploy-azure-function)
5. [Deploy to Nomad](#deploy-to-nomad)
6. [Configure CI/CD](#configure-cicd)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Tools

Install these tools on your development machine:

```bash
# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Terraform (>=1.5.0)
wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
unzip terraform_1.5.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Nomad CLI (optional, for management)
wget https://releases.hashicorp.com/nomad/1.7.2/nomad_1.7.2_linux_amd64.zip
unzip nomad_1.7.2_linux_amd64.zip
sudo mv nomad /usr/local/bin/

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### Azure Account

- **Azure Free Trial**: [Sign up here](https://azure.microsoft.com/free/)
- $200 credit for 30 days
- Free tier services (Functions, Blob Storage)

### SSH Key

Generate SSH key pair if you don't have one:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/fiap_azure_rsa
# Public key: ~/.ssh/fiap_azure_rsa.pub
# Private key: ~/.ssh/fiap_azure_rsa
```

---

## Initial Setup

### 1. Login to Azure

```bash
az login
```

### 2. Set Subscription

```bash
# List subscriptions
az account list --output table

# Set active subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### 3. Create Service Principal for Terraform

```bash
# Create service principal
az ad sp create-for-rbac \
  --name "fiap-challenge-terraform" \
  --role="Contributor" \
  --scopes="/subscriptions/YOUR_SUBSCRIPTION_ID" \
  --sdk-auth > azure-credentials.json

# Extract values
export ARM_CLIENT_ID=$(cat azure-credentials.json | jq -r .clientId)
export ARM_CLIENT_SECRET=$(cat azure-credentials.json | jq -r .clientSecret)
export ARM_SUBSCRIPTION_ID=$(cat azure-credentials.json | jq -r .subscriptionId)
export ARM_TENANT_ID=$(cat azure-credentials.json | jq -r .tenantId)
```

⚠️ **IMPORTANT**: Keep `azure-credentials.json` secure and never commit to Git!

### 4. Clone Repository

```bash
git clone https://github.com/vnzoliveira/STDR.DataScience.PJlook.git
cd STDR.DataScience.PJlook
```

---

## Deploy Infrastructure

### 1. Configure Terraform Variables

```bash
cd terraform/

# Copy example file
cp terraform.tfvars.example terraform.tfvars

# Edit with your values
nano terraform.tfvars
```

**terraform.tfvars**:
```hcl
project_name = "fiapchallenge"
environment  = "dev"
location     = "brazilsouth"

nomad_vm_count = 1  # Start with 1 for testing
nomad_vm_size  = "Standard_B2ms"

admin_username       = "azureuser"
admin_ssh_public_key = "ssh-rsa AAAAB3... your-public-key-here"

enable_auto_shutdown = true
auto_shutdown_time   = "1900"
```

### 2. Initialize Terraform

```bash
terraform init
```

### 3. Plan Infrastructure

```bash
terraform plan -var-file=environments/dev.tfvars
```

Review the plan carefully. Expected resources:
- 1-3 VMs for Nomad cluster
- Storage Account with 3 containers
- Azure Function App
- Key Vault
- Virtual Network & NSG
- Load Balancer

### 4. Apply Infrastructure

```bash
terraform apply -var-file=environments/dev.tfvars

# Type 'yes' when prompted
```

⏱️ **Time**: ~15-20 minutes

### 5. Save Outputs

```bash
# Get all outputs
terraform output

# Save specific outputs
terraform output storage_connection_string > ../secrets/storage-conn.txt
terraform output nomad_vm_public_ips
terraform output dashboard_url
```

---

## Deploy Azure Function

### 1. Navigate to Function Directory

```bash
cd ../azure-functions/
```

### 2. Test Locally (Optional)

```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Install dependencies
pip install -r requirements.txt

# Copy settings
cp local.settings.json.example local.settings.json
# Edit with your storage connection string

# Run locally
func start
```

### 3. Deploy to Azure

```bash
# Get Function App name from Terraform
FUNC_APP_NAME=$(cd ../terraform && terraform output -raw function_app_name)

# Deploy
func azure functionapp publish $FUNC_APP_NAME --python
```

### 4. Verify Deployment

```bash
# Test health endpoint
FUNC_URL=$(cd ../terraform && terraform output -raw function_app_url)
curl $FUNC_URL/api/health
```

---

## Deploy to Nomad

### 1. Connect to Nomad Server

```bash
# Get Nomad VM IP
NOMAD_IP=$(cd terraform && terraform output -json nomad_vm_public_ips | jq -r '.[0]')

# SSH to Nomad server
ssh -i ~/.ssh/fiap_azure_rsa azureuser@$NOMAD_IP
```

### 2. Verify Nomad is Running

```bash
sudo systemctl status nomad
nomad server members
nomad node status
```

### 3. Configure Nomad Variables

On the Nomad server, set secrets:

```bash
# Storage credentials
nomad var put azure/storage_account "your-storage-account-name"
nomad var put azure/storage_key "your-storage-key"
nomad var put azure/key_vault_uri "https://your-keyvault.vault.azure.net/"

# Docker registry credentials
nomad var put docker/username "your-github-username"
nomad var put docker/password "your-github-personal-access-token"
```

### 4. Update Nomad Job Files

On your local machine, update Docker image references:

```bash
cd ../nomad/

# Replace placeholders with your values
sed -i "s/YOUR_GITHUB_USERNAME/vnzoliveira/g" *.nomad
sed -i "s/REPO_NAME/STDR.DataScience.PJlook/g" *.nomad
```

### 5. Deploy Jobs to Nomad

```bash
# Set Nomad address
export NOMAD_ADDR=http://$NOMAD_IP:4646

# Deploy ML Pipeline
nomad job run ml-pipeline.nomad

# Deploy Dashboard
nomad job run dashboard.nomad
```

### 6. Verify Deployments

```bash
# Check job status
nomad job status ml-pipeline
nomad job status dashboard

# View logs
nomad alloc logs -f $(nomad job allocs dashboard -json | jq -r '.[0].ID')
```

---

## Configure CI/CD

### 1. Setup GitHub Secrets

Go to your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

| Secret Name | Value | How to Get |
|------------|-------|------------|
| `AZURE_CREDENTIALS` | Content of `azure-credentials.json` | From service principal creation |
| `AZURE_FUNCTION_APP_NAME` | Function app name | `terraform output function_app_name` |
| `NOMAD_ADDR` | `http://YOUR_NOMAD_IP:4646` | From Terraform output |
| `NOMAD_TOKEN` | Nomad ACL token (if enabled) | `nomad acl bootstrap` |
| `ARM_CLIENT_ID` | Azure client ID | From service principal |
| `ARM_CLIENT_SECRET` | Azure client secret | From service principal |
| `ARM_SUBSCRIPTION_ID` | Azure subscription ID | `az account show` |
| `ARM_TENANT_ID` | Azure tenant ID | From service principal |
| `SSH_PUBLIC_KEY` | Your SSH public key | Content of `~/.ssh/fiap_azure_rsa.pub` |

### 2. Test CI/CD Pipeline

```bash
# Make a test commit
git checkout -b test/ci-cd
echo "# Test CI/CD" >> README.md
git add README.md
git commit -m "test: CI/CD pipeline"
git push origin test/ci-cd

# Create pull request and watch GitHub Actions
```

### 3. Deploy to Production

```bash
# Merge to main branch
git checkout main
git merge test/ci-cd
git push origin main

# Watch deployment in GitHub Actions
```

---

## Verification

### 1. Test Azure Function

Upload a test Excel file:

```bash
# Get storage account name
STORAGE_ACCOUNT=$(cd terraform && terraform output -raw storage_account_name)

# Upload test file
az storage blob upload \
  --account-name $STORAGE_ACCOUNT \
  --container-name raw-data \
  --name test.xlsx \
  --file data/raw/Challenge_FIAP_Bases.xlsx

# Check if parquet files were created
az storage blob list \
  --account-name $STORAGE_ACCOUNT \
  --container-name processed-data \
  --output table
```

### 2. Test ML Pipeline

```bash
# Trigger pipeline manually
nomad job dispatch ml-pipeline

# Watch logs
nomad alloc logs -f $(nomad job allocs ml-pipeline -json | jq -r '.[0].ID')
```

### 3. Access Dashboard

```bash
# Get dashboard URL
DASHBOARD_URL=$(cd terraform && terraform output -raw dashboard_url)

# Open in browser
echo "Dashboard: $DASHBOARD_URL"
```

---

## Cost Monitoring

### Check Current Costs

```bash
# View cost analysis
az consumption usage list \
  --start-date 2025-10-01 \
  --end-date 2025-10-11 \
  --query "[?contains(instanceName, 'fiapchallenge')]" \
  --output table
```

### Stop Resources to Save Money

```bash
# Stop VMs (nights/weekends)
az vm deallocate \
  --resource-group fiapchallenge-dev \
  --name fiapchallenge-dev-nomad-vm-0

# Restart when needed
az vm start \
  --resource-group fiapchallenge-dev \
  --name fiapchallenge-dev-nomad-vm-0
```

### Destroy Everything

⚠️ **WARNING**: This deletes all resources!

```bash
cd terraform/
terraform destroy -var-file=environments/dev.tfvars

# Type 'yes' to confirm
```

---

## Troubleshooting

### Terraform Errors

**Problem**: `Error creating Resource Group`

**Solution**:
```bash
# Ensure logged in
az login

# Check permissions
az role assignment list --assignee $ARM_CLIENT_ID
```

### Function Deployment Fails

**Problem**: Function not responding

**Solution**:
```bash
# Check function logs
func azure functionapp logstream $FUNC_APP_NAME

# Restart function
az functionapp restart --name $FUNC_APP_NAME --resource-group RESOURCE_GROUP
```

### Nomad Jobs Not Starting

**Problem**: Docker images not pulling

**Solution**:
```bash
# On Nomad VM
ssh azureuser@$NOMAD_IP

# Try pulling manually
docker login ghcr.io -u USERNAME -p TOKEN
docker pull ghcr.io/vnzoliveira/STDR.DataScience.PJlook/dashboard:latest

# Check Nomad logs
sudo journalctl -u nomad -f
```

### Dashboard Not Accessible

**Problem**: Cannot access dashboard URL

**Solution**:
```bash
# Check NSG rules
az network nsg rule list \
  --resource-group fiapchallenge-dev \
  --nsg-name fiapchallenge-dev-nsg \
  --output table

# Check if dashboard is running
nomad job status dashboard

# Check from VM directly
ssh azureuser@$NOMAD_IP
curl localhost:8060
```

---

## Next Steps

1. **Enable Monitoring**: Setup Azure Monitor and Application Insights
2. **Configure Alerts**: Set up cost alerts and health alerts
3. **Backup Strategy**: Configure blob storage backups
4. **Scale Up**: Move from dev (1 VM) to prod (3 VMs)
5. **Security**: Enable Azure Key Vault for secrets, configure network policies

---

## Support

- **GitHub Issues**: [Create an issue](https://github.com/vnzoliveira/STDR.DataScience.PJlook/issues)
- **Documentation**: [Main README](../README.md)
- **Azure Docs**: [Azure Documentation](https://docs.microsoft.com/azure/)


