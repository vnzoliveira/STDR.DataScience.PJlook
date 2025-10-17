# FIAP Challenge - Azure Cloud Deployment

<div align="center">

[![CI/CD](https://github.com/vnzoliveira/STDR.DataScience.PJlook/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/vnzoliveira/STDR.DataScience.PJlook/actions)
[![Azure](https://img.shields.io/badge/Azure-0078D4?logo=microsoft-azure&logoColor=white)](https://portal.azure.com)
[![Terraform](https://img.shields.io/badge/Terraform-7B42BC?logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Machine Learning Platform for Company Stage Classification and Risk Analysis**

[Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture) • [Deployment](#-deployment)

</div>

---

## 📋 Overview

Production-ready data science platform deployed on **Azure Cloud** using infrastructure-as-code (Terraform), containerized workloads (Docker), and orchestration (Nomad). Features automated ETL, ensemble ML models, and interactive dashboards.

### Key Features

✅ **Automated ETL** - Azure Functions for xlsx→parquet conversion  
✅ **ML Pipeline** - Ensemble models with Random Forest & Gradient Boosting  
✅ **Social Network Analysis** - Company relationship mapping  
✅ **Risk Scoring** - Multi-factor risk assessment  
✅ **Interactive Dashboard** - Dash-based visualization  
✅ **CI/CD** - Automated deployment via GitHub Actions  
✅ **Infrastructure as Code** - Full Terraform configuration  
✅ **Cost Optimized** - Runs on Azure free tier ($200 credit)  

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required tools
- Azure CLI
- Terraform >= 1.5.0
- Docker & Docker Compose
- Python 3.11+
- Git
```

### Deploy in 3 Commands

```bash
# 1. Clone and navigate
git clone https://github.com/vnzoliveira/STDR.DataScience.PJlook.git
cd STDR.DataScience.PJlook

# 2. Deploy infrastructure
cd terraform
terraform init
terraform apply -var-file=environments/dev.tfvars -auto-approve

# 3. Access dashboard
terraform output dashboard_url
```

**Time**: ~30 minutes | **Cost**: ~$30-40/month (5-6 months with free $200 credit)

👉 **Detailed Guide**: [QUICK_START.md](docs/QUICK_START.md)

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [**Quick Start**](docs/QUICK_START.md) | Get running in 30 minutes |
| [**Deployment Guide**](docs/DEPLOYMENT_GUIDE.md) | Complete deployment walkthrough |
| [**Architecture**](docs/ARCHITECTURE.md) | System design and components |
| [**Nomad Jobs**](nomad/README.md) | Job definitions and management |

---

## 🏗️ Architecture

```
┌─────────────────── Azure Cloud (Brazil South) ──────────────────┐
│                                                                   │
│  ┌──────────────┐   ┌─────────────┐   ┌──────────────────────┐ │
│  │ Blob Storage │   │  Key Vault  │   │ Azure Functions      │ │
│  │ Data Lake    │   │  Secrets    │   │ xlsx → parquet       │ │
│  └──────┬───────┘   └──────┬──────┘   └──────────┬───────────┘ │
│         │                   │                      │             │
│  ───────┴───────────────────┴──────────────────────┴──────────  │
│                                                                   │
│  ┌────────────── Nomad Cluster (VMs) ───────────────────────┐  │
│  │                                                            │  │
│  │  ┌───────────────────┐      ┌─────────────────────────┐  │  │
│  │  │  ML Pipeline      │      │  Dashboard              │  │  │
│  │  │  (Batch - Daily)  │      │  (Service - 24/7)       │  │  │
│  │  │                   │      │                         │  │  │
│  │  │  • Data Ingestion │      │  • Dash Web App         │  │  │
│  │  │  • Feature Eng    │      │  • Plotly Charts        │  │  │
│  │  │  • ML Models      │      │  • Network Viz          │  │  │
│  │  │  • SNA Analysis   │      │  • Risk Dashboard       │  │  │
│  │  │  • Risk Scoring   │      │                         │  │  │
│  │  └───────────────────┘      └─────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                            ↑                                      │
│                     ┌──────┴───────┐                             │
│                     │Load Balancer │ → 🌐 Public Access          │
│                     └──────────────┘                             │
└───────────────────────────────────────────────────────────────────┘
          ↑
    ┌─────┴──────┐
    │  CI/CD     │  GitHub Actions
    │  Pipeline  │  • Build • Test • Deploy
    └────────────┘
```

---

## 💻 Technology Stack

### Infrastructure
- **Cloud**: Azure (Brazil South region)
- **IaC**: Terraform
- **Orchestration**: HashiCorp Nomad
- **Containers**: Docker
- **CI/CD**: GitHub Actions

### Data & ML
- **ETL**: Pandas, PyArrow
- **Storage**: Azure Blob Storage (Parquet)
- **ML**: scikit-learn (Random Forest, Gradient Boosting)
- **SNA**: NetworkX
- **Query**: DuckDB

### Application
- **Dashboard**: Dash (Plotly)
- **Visualization**: Plotly, Dash Cytoscape
- **UI**: Dash Bootstrap Components

---

## 📊 Components

### 1. Azure Function (Serverless ETL)
- **Trigger**: Blob upload to `raw-data` container
- **Process**: Excel → Parquet conversion
- **Runtime**: Python 3.11
- **Cost**: Free (within 1M executions/month)

### 2. ML Pipeline (Nomad Batch Job)
- **Schedule**: Daily at 2 AM
- **Models**: 
  - Simple rule-based classifier
  - Unsupervised clustering
  - Supervised ML (RF + GB)
  - Ensemble model
- **Analysis**: 
  - Social Network Analysis (SNA)
  - Sector benchmarking
  - Risk scoring
- **Resources**: 4 CPU cores, 8GB RAM

### 3. Dashboard (Nomad Service)
- **Port**: 8060
- **Access**: Via load balancer or direct VM IP
- **Pages**:
  - Desafio 1: Company profiles & stage classification
  - Desafio 2: Relationship networks & risk analysis
- **Resources**: 1 CPU core, 2GB RAM

---

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python -m src.cli.main build-all

# Start dashboard
python -m src.ui.app
```

### Docker

```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Run pipeline
docker-compose --profile pipeline up

# Run dashboard
docker-compose up dashboard
```

### Azure Cloud

#### 1. Setup

```bash
# Login
az login

# Navigate to Terraform
cd terraform/

# Configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

#### 2. Deploy Infrastructure

```bash
# Initialize
terraform init

# Review plan
terraform plan -var-file=environments/dev.tfvars

# Deploy
terraform apply -var-file=environments/dev.tfvars
```

#### 3. Deploy Azure Function

```bash
cd ../azure-functions
FUNC_APP=$(cd ../terraform && terraform output -raw function_app_name)
func azure functionapp publish $FUNC_APP --python
```

#### 4. Deploy Nomad Jobs

```bash
# Get Nomad address
export NOMAD_ADDR=http://$(cd ../terraform && terraform output -json nomad_vm_public_ips | jq -r '.[0]'):4646

# Configure Nomad variables
nomad var put azure/storage_account "your-account"
nomad var put azure/storage_key "your-key"

# Deploy jobs
cd ../nomad
nomad job run ml-pipeline.nomad
nomad job run dashboard.nomad
```

#### 5. Access Dashboard

```bash
# Get URL
cd ../terraform
terraform output dashboard_url
```

---

## 💰 Cost Breakdown

### Development Environment (1 VM)

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Blob Storage | $2-3 | Hot tier |
| Azure Functions | $0-1 | Free tier |
| Key Vault | <$1 | Low usage |
| VM (B2ms) | $30-35 | 2 vCPU, 8GB |
| Load Balancer | $20-25 | Standard SKU |
| **Total** | **~$55-60** | |

**With Auto-Shutdown**: ~$30-40/month  
**Free Trial**: Runs 5-6 months on $200 credit

### Production Environment (3 VMs)

| Component | Monthly Cost |
|-----------|--------------|
| Infrastructure | ~$120-150 |
| With Reserved Instances | ~$85-100 (40% savings) |
| With Spot VMs | ~$50-75 (60-90% savings) |

---

## 🔒 Security

### Authentication
- ✅ SSH key-based access (no passwords)
- ✅ Azure managed identities
- ✅ Service principal for CI/CD

### Secrets Management
- ✅ Azure Key Vault for all secrets
- ✅ No secrets in code or logs
- ✅ Automated secret rotation (recommended)

### Network Security
- ✅ NSG firewall rules
- ✅ Private virtual network
- ⚠️ TODO: HTTPS/SSL certificates
- ⚠️ TODO: WAF implementation

---

## 📈 Monitoring

### Current

```bash
# Function logs
func azure functionapp logstream $FUNC_APP_NAME

# Nomad jobs
nomad alloc logs <alloc-id>

# VM logs
ssh azureuser@$VM_IP
sudo journalctl -u nomad -f
```

### Recommended (Future)
- Azure Monitor + Application Insights
- Grafana + Prometheus dashboards
- Cost alerts and budgets

---

## 🧪 Testing

### Run Tests Locally

```bash
# Install dev dependencies
pip install pytest pytest-cov ruff

# Lint
ruff check src/

# Test
pytest tests/ --cov=src
```

### Test in CI/CD

- Automated on every push
- See `.github/workflows/ci-cd.yml`
- View results in GitHub Actions tab

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- **FIAP** - For the challenge and opportunity
- **Santander** - For sponsorship and data
- **HashiCorp** - For Nomad and Terraform
- **Microsoft Azure** - For cloud infrastructure
- **Plotly** - For Dash framework

---

## 📞 Support

### Get Help

- 📚 [Documentation](docs/)
- 🐛 [Report Bug](https://github.com/vnzoliveira/STDR.DataScience.PJlook/issues)
- 💡 [Request Feature](https://github.com/vnzoliveira/STDR.DataScience.PJlook/issues)
- 💬 [Discussions](https://github.com/vnzoliveira/STDR.DataScience.PJlook/discussions)

### Maintainers

- [@vnzoliveira](https://github.com/vnzoliveira)

---

## 📊 Project Status

- ✅ Phase 1: Local development - Complete
- ✅ Phase 2: Dockerization - Complete
- ✅ Phase 3: Cloud infrastructure - Complete
- ✅ Phase 4: CI/CD automation - Complete
- 🔄 Phase 5: Production hardening - In Progress
- ⏳ Phase 6: Advanced monitoring - Planned

---

<div align="center">

**Made with ❤️ for the FIAP Datalab Santander Challenge**

[⬆ Back to Top](#fiap-challenge---azure-cloud-deployment)

</div>


