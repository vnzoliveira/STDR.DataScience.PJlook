# Quick Start Guide

Get the FIAP Challenge app running in 30 minutes!

## 🚀 Fastest Path to Running App

### Option 1: Local Development (No Cloud Required)

```bash
# 1. Clone repository
git clone https://github.com/vnzoliveira/STDR.DataScience.PJlook.git
cd STDR.DataScience.PJlook

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ETL pipeline
python -m src.cli.main build-all

# 4. Start dashboard
python -m src.ui.app

# 5. Open browser
# Navigate to http://localhost:8060
```

**Time**: ~10 minutes

---

### Option 2: Docker (Recommended for Testing)

```bash
# 1. Clone repository
git clone https://github.com/vnzoliveira/STDR.DataScience.PJlook.git
cd STDR.DataScience.PJlook

# 2. Build images
docker-compose -f docker/docker-compose.yml build

# 3. Run pipeline
docker-compose -f docker/docker-compose.yml --profile pipeline up

# 4. Start dashboard
docker-compose -f docker/docker-compose.yml up dashboard

# 5. Open browser
# Navigate to http://localhost:8060
```

**Time**: ~15 minutes

---

### Option 3: Azure Free Tier (Production-Ready)

#### Prerequisites
- Azure free trial account ($200 credit)
- Azure CLI installed
- SSH key generated

#### Steps

```bash
# 1. Login to Azure
az login

# 2. Navigate to terraform directory
cd terraform/

# 3. Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_name = "fiapchallenge"
environment  = "dev"
location     = "brazilsouth"
nomad_vm_count = 1
nomad_vm_size = "Standard_B2ms"
admin_username = "azureuser"
admin_ssh_public_key = "$(cat ~/.ssh/id_rsa.pub)"
EOF

# 4. Deploy infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars -auto-approve

# 5. Deploy Azure Function
cd ../azure-functions
FUNC_APP=$(cd ../terraform && terraform output -raw function_app_name)
func azure functionapp publish $FUNC_APP --python

# 6. Get dashboard URL
cd ../terraform
terraform output dashboard_url

# 7. Open browser to dashboard URL
```

**Time**: ~30 minutes  
**Cost**: ~$30-40/month (~5-6 months with $200 credit)

---

## 📊 What You Get

After deployment:

1. **Azure Function**: Automatically converts `.xlsx` → `.parquet`
2. **ML Pipeline**: Runs daily at 2 AM
   - Ensemble models
   - Supervised/unsupervised classification
   - Social network analysis
   - Risk scoring
3. **Dashboard**: Interactive Dash application
   - Company profiles
   - Stage classification
   - Relationship network visualization
   - Sector analysis

---

## 🎯 Testing the System

### Upload Test Data

```bash
# Get storage account
STORAGE=$(cd terraform && terraform output -raw storage_account_name)

# Upload Excel file
az storage blob upload \
  --account-name $STORAGE \
  --container-name raw-data \
  --name test.xlsx \
  --file data/raw/Challenge_FIAP_Bases.xlsx
```

**Result**: Function automatically processes → Creates parquet files

### Trigger ML Pipeline

```bash
# Get Nomad address
export NOMAD_ADDR=http://$(cd terraform && terraform output -json nomad_vm_public_ips | jq -r '.[0]'):4646

# Run pipeline manually
nomad job dispatch ml-pipeline

# Watch progress
nomad alloc logs -f $(nomad job allocs ml-pipeline -json | jq -r '.[0].ID')
```

**Result**: Generates model outputs in `reports/exports/`

### View Dashboard

```bash
# Get URL
DASHBOARD_URL=$(cd terraform && terraform output -raw dashboard_url)
echo "Dashboard: $DASHBOARD_URL"

# Or access via Nomad VM
NOMAD_IP=$(cd terraform && terraform output -json nomad_vm_public_ips | jq -r '.[0]')
echo "Dashboard: http://$NOMAD_IP:8060"
```

---

## 💰 Cost Control

### Save Money Tips

1. **Deallocate VMs when not in use**:
   ```bash
   az vm deallocate --resource-group fiapchallenge-dev --name fiapchallenge-dev-nomad-vm-0
   ```

2. **Use auto-shutdown** (already configured for dev):
   - VMs automatically stop at 7 PM BRT
   - Manual restart: `az vm start ...`

3. **Use Spot VMs** for 60-90% discount:
   - Edit `terraform/variables.tf`
   - Add spot VM configuration

4. **Monitor costs**:
   ```bash
   az consumption usage list --start-date 2025-10-01 --end-date 2025-10-11
   ```

### Expected Monthly Costs

| Environment | Components | Monthly Cost |
|------------|-----------|--------------|
| **Dev** (1 VM) | VM + Storage + Function | ~$35-40 |
| **Staging** (2 VMs) | VM + Storage + Function | ~$75-85 |
| **Prod** (3 VMs) | VM + Storage + Function | ~$120-150 |

**Free Tier Components** (always free):
- Azure Functions: 1M executions/month
- Blob Storage: First 5GB
- Bandwidth: First 15GB/month

---

## 🐛 Troubleshooting

### Problem: Terraform fails with permission error

**Solution**:
```bash
# Check Azure login
az account show

# Ensure correct subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verify service principal
az ad sp show --id $ARM_CLIENT_ID
```

### Problem: Dashboard not accessible

**Solution**:
```bash
# Check if VM is running
az vm get-instance-view --resource-group fiapchallenge-dev --name fiapchallenge-dev-nomad-vm-0 | jq .statuses

# Check Nomad job
export NOMAD_ADDR=http://$NOMAD_IP:4646
nomad job status dashboard

# Check NSG rules
az network nsg rule list --resource-group fiapchallenge-dev --nsg-name fiapchallenge-dev-nsg
```

### Problem: Function not triggering

**Solution**:
```bash
# Check function logs
func azure functionapp logstream $FUNC_APP_NAME

# Test manually
curl https://$FUNC_APP_NAME.azurewebsites.net/api/health

# Check blob trigger
az storage blob list --account-name $STORAGE --container-name raw-data
```

---

## 📚 Next Steps

1. **Explore Dashboard**: Navigate through different pages
2. **Run Pipeline**: Watch ML models train
3. **Upload Data**: Test with your own Excel files
4. **Configure CI/CD**: Setup GitHub Actions (see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md))
5. **Scale Up**: Move to 3 VMs for high availability

---

## 📖 Full Documentation

- **Complete Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [API.md](API.md)
- **Development**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🆘 Get Help

- **GitHub Issues**: [Report a bug](https://github.com/vnzoliveira/STDR.DataScience.PJlook/issues)
- **Discussions**: [Ask a question](https://github.com/vnzoliveira/STDR.DataScience.PJlook/discussions)
- **Email**: Contact the maintainers

---

**Happy Deploying! 🚀**


