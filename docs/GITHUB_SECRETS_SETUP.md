# GitHub Secrets Setup Guide

Complete guide to configure GitHub Actions secrets for CI/CD pipeline.

## 📋 Required Secrets

Your repository needs the following secrets configured for automated deployments:

### Azure Credentials

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `AZURE_CREDENTIALS` | Service principal credentials (JSON) | See [Step 1](#step-1-create-service-principal) |
| `ARM_CLIENT_ID` | Azure AD application ID | From service principal output |
| `ARM_CLIENT_SECRET` | Application secret | From service principal output |
| `ARM_SUBSCRIPTION_ID` | Azure subscription ID | `az account show --query id -o tsv` |
| `ARM_TENANT_ID` | Azure AD tenant ID | From service principal output |

### Azure Resources

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `AZURE_FUNCTION_APP_NAME` | Function app name | `terraform output function_app_name` |
| `AZURE_STORAGE_ACCOUNT` | Storage account name | `terraform output storage_account_name` |

### Nomad

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `NOMAD_ADDR` | Nomad server address | `http://YOUR_VM_IP:4646` |
| `NOMAD_TOKEN` | Nomad ACL token | `nomad acl bootstrap` (if ACL enabled) |

### SSH Access

| Secret Name | Description | How to Get |
|------------|-------------|------------|
| `SSH_PUBLIC_KEY` | SSH public key | Content of `~/.ssh/id_rsa.pub` |
| `SSH_PRIVATE_KEY` | SSH private key (optional) | Content of `~/.ssh/id_rsa` |

---

## 🔐 Step-by-Step Setup

### Step 1: Create Service Principal

Open terminal and run:

```bash
# Login to Azure
az login

# Get subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"

# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-fiap" \
  --role Contributor \
  --scopes "/subscriptions/$SUBSCRIPTION_ID" \
  --sdk-auth > azure-credentials.json

# Display credentials
cat azure-credentials.json
```

**Output** (example):
```json
{
  "clientId": "12345678-1234-1234-1234-123456789012",
  "clientSecret": "your-secret-here",
  "subscriptionId": "87654321-4321-4321-4321-210987654321",
  "tenantId": "abcdefgh-abcd-abcd-abcd-abcdefghijkl",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

⚠️ **Important**: Save this file securely and never commit to Git!

### Step 2: Extract Values

```bash
# Extract individual values
export ARM_CLIENT_ID=$(cat azure-credentials.json | jq -r .clientId)
export ARM_CLIENT_SECRET=$(cat azure-credentials.json | jq -r .clientSecret)
export ARM_SUBSCRIPTION_ID=$(cat azure-credentials.json | jq -r .subscriptionId)
export ARM_TENANT_ID=$(cat azure-credentials.json | jq -r .tenantId)

echo "CLIENT_ID: $ARM_CLIENT_ID"
echo "SUBSCRIPTION_ID: $ARM_SUBSCRIPTION_ID"
echo "TENANT_ID: $ARM_TENANT_ID"
echo "SECRET: [hidden]"
```

### Step 3: Deploy Infrastructure (if not done yet)

```bash
cd terraform/

# Initialize
terraform init

# Deploy
terraform apply -var-file=environments/dev.tfvars

# Get outputs
FUNC_APP_NAME=$(terraform output -raw function_app_name)
STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)
NOMAD_IP=$(terraform output -json nomad_vm_public_ips | jq -r '.[0]')

echo "Function App: $FUNC_APP_NAME"
echo "Storage Account: $STORAGE_ACCOUNT"
echo "Nomad IP: $NOMAD_IP"
```

### Step 4: Setup Nomad (Optional)

If you want to enable Nomad ACL for security:

```bash
# SSH to Nomad server
ssh azureuser@$NOMAD_IP

# Bootstrap ACL
nomad acl bootstrap > /tmp/nomad-token.txt

# Get management token
NOMAD_TOKEN=$(cat /tmp/nomad-token.txt | grep "Secret ID" | awk '{print $4}')
echo "Nomad Token: $NOMAD_TOKEN"
```

### Step 5: Get SSH Keys

```bash
# Generate new key if needed
ssh-keygen -t rsa -b 4096 -f ~/.ssh/fiap_azure_rsa

# Display public key
cat ~/.ssh/fiap_azure_rsa.pub

# Display private key (optional, for automated SSH)
cat ~/.ssh/fiap_azure_rsa
```

---

## 🔧 Configure GitHub Secrets

### Via GitHub Web UI

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret:

#### Add `AZURE_CREDENTIALS`
- Name: `AZURE_CREDENTIALS`
- Value: Paste entire content of `azure-credentials.json`

#### Add `ARM_CLIENT_ID`
- Name: `ARM_CLIENT_ID`
- Value: `12345678-1234-1234-1234-123456789012`

#### Add `ARM_CLIENT_SECRET`
- Name: `ARM_CLIENT_SECRET`
- Value: `your-secret-here`

#### Add `ARM_SUBSCRIPTION_ID`
- Name: `ARM_SUBSCRIPTION_ID`
- Value: `87654321-4321-4321-4321-210987654321`

#### Add `ARM_TENANT_ID`
- Name: `ARM_TENANT_ID`
- Value: `abcdefgh-abcd-abcd-abcd-abcdefghijkl`

#### Add `AZURE_FUNCTION_APP_NAME`
- Name: `AZURE_FUNCTION_APP_NAME`
- Value: From `terraform output function_app_name`

#### Add `AZURE_STORAGE_ACCOUNT`
- Name: `AZURE_STORAGE_ACCOUNT`
- Value: From `terraform output storage_account_name`

#### Add `NOMAD_ADDR`
- Name: `NOMAD_ADDR`
- Value: `http://YOUR_VM_IP:4646`

#### Add `NOMAD_TOKEN` (Optional)
- Name: `NOMAD_TOKEN`
- Value: From `nomad acl bootstrap` output

#### Add `SSH_PUBLIC_KEY`
- Name: `SSH_PUBLIC_KEY`
- Value: Content of `~/.ssh/fiap_azure_rsa.pub`

### Via GitHub CLI

```bash
# Install GitHub CLI if needed
# https://cli.github.com/

# Login
gh auth login

# Set repository
REPO="vnzoliveira/STDR.DataScience.PJlook"

# Add secrets
gh secret set AZURE_CREDENTIALS -b "$(cat azure-credentials.json)" -R $REPO
gh secret set ARM_CLIENT_ID -b "$ARM_CLIENT_ID" -R $REPO
gh secret set ARM_CLIENT_SECRET -b "$ARM_CLIENT_SECRET" -R $REPO
gh secret set ARM_SUBSCRIPTION_ID -b "$ARM_SUBSCRIPTION_ID" -R $REPO
gh secret set ARM_TENANT_ID -b "$ARM_TENANT_ID" -R $REPO
gh secret set AZURE_FUNCTION_APP_NAME -b "$FUNC_APP_NAME" -R $REPO
gh secret set AZURE_STORAGE_ACCOUNT -b "$STORAGE_ACCOUNT" -R $REPO
gh secret set NOMAD_ADDR -b "http://$NOMAD_IP:4646" -R $REPO
gh secret set SSH_PUBLIC_KEY -b "$(cat ~/.ssh/fiap_azure_rsa.pub)" -R $REPO

echo "✅ All secrets configured!"
```

---

## ✅ Verify Configuration

### Check Secrets are Set

```bash
# List configured secrets
gh secret list -R $REPO
```

Expected output:
```
AZURE_CREDENTIALS         Updated 2025-10-11
ARM_CLIENT_ID             Updated 2025-10-11
ARM_CLIENT_SECRET         Updated 2025-10-11
ARM_SUBSCRIPTION_ID       Updated 2025-10-11
ARM_TENANT_ID             Updated 2025-10-11
AZURE_FUNCTION_APP_NAME   Updated 2025-10-11
AZURE_STORAGE_ACCOUNT     Updated 2025-10-11
NOMAD_ADDR                Updated 2025-10-11
SSH_PUBLIC_KEY            Updated 2025-10-11
```

### Test CI/CD Pipeline

```bash
# Create test branch
git checkout -b test/ci-cd-setup

# Make a small change
echo "# CI/CD Test" >> .github/test.txt
git add .github/test.txt
git commit -m "test: verify CI/CD pipeline"

# Push and watch
git push origin test/ci-cd-setup

# Check GitHub Actions
# Go to: https://github.com/$REPO/actions
```

---

## 🔄 Update Secrets

### When to Update

- **Service Principal**: Expires after 1-2 years
- **Nomad Token**: When regenerated or ACL reset
- **SSH Keys**: When rotated for security
- **Storage Keys**: When rotated

### How to Update

#### Service Principal

```bash
# Reset credentials
az ad sp credential reset \
  --name "github-actions-fiap" \
  --sdk-auth > azure-credentials-new.json

# Update GitHub secret
gh secret set AZURE_CREDENTIALS -b "$(cat azure-credentials-new.json)" -R $REPO
gh secret set ARM_CLIENT_SECRET -b "$(cat azure-credentials-new.json | jq -r .clientSecret)" -R $REPO
```

#### Nomad Token

```bash
# SSH to Nomad
ssh azureuser@$NOMAD_IP

# Create new token
nomad acl token create -name="github-actions" -policy="deploy"

# Update secret
gh secret set NOMAD_TOKEN -b "NEW_TOKEN_HERE" -R $REPO
```

---

## 🔒 Security Best Practices

### ✅ Do

- Rotate secrets regularly (every 90 days)
- Use service principals with minimal permissions
- Enable Azure Key Vault for sensitive data
- Monitor secret usage in GitHub Actions logs
- Use environment protection rules for production

### ❌ Don't

- Never commit secrets to Git
- Don't share `azure-credentials.json`
- Don't log secret values in workflows
- Don't use personal accounts for service principals
- Don't grant more permissions than needed

### Secure Storage

```bash
# Store secrets securely on your machine
mkdir -p ~/.azure/secrets
chmod 700 ~/.azure/secrets

mv azure-credentials.json ~/.azure/secrets/
chmod 600 ~/.azure/secrets/azure-credentials.json

# Use password manager for long-term storage
# Recommended: 1Password, Bitwarden, Azure Key Vault
```

---

## 🐛 Troubleshooting

### Secret Not Found Error

**Problem**: Workflow fails with "Secret not found"

**Solution**:
```bash
# Check secret name (case-sensitive!)
gh secret list -R $REPO

# Verify in workflow file
grep "secrets\." .github/workflows/ci-cd.yml
```

### Authentication Failed

**Problem**: Azure login fails in CI/CD

**Solution**:
```bash
# Test credentials locally
az login --service-principal \
  -u $ARM_CLIENT_ID \
  -p $ARM_CLIENT_SECRET \
  --tenant $ARM_TENANT_ID

# If successful, re-add to GitHub
gh secret set AZURE_CREDENTIALS -b "$(cat azure-credentials.json)" -R $REPO
```

### Nomad Connection Refused

**Problem**: Cannot connect to Nomad from CI/CD

**Solution**:
```bash
# Check VM is running
az vm get-instance-view \
  --resource-group fiapchallenge-dev \
  --name fiapchallenge-dev-nomad-vm-0 \
  --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" -o tsv

# Check NSG allows port 4646
az network nsg rule show \
  --resource-group fiapchallenge-dev \
  --nsg-name fiapchallenge-dev-nsg \
  --name Nomad-HTTP

# Test from your machine
curl http://$NOMAD_IP:4646/v1/status/leader
```

---

## 📚 References

- [GitHub Actions Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [Azure Service Principals](https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal)
- [Nomad ACL System](https://www.nomadproject.io/docs/configuration/acl)

---

**✅ Setup Complete!** Your CI/CD pipeline is now configured and ready to deploy.


