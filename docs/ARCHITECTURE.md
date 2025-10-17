# Architecture Documentation

## System Overview

The FIAP Challenge application is a cloud-native data science platform deployed on Azure, using a microservices architecture with Nomad for orchestration.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FIAP Challenge Platform                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Azure Cloud (Brazil South)             │  │
│  │                                                            │  │
│  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │ Blob Storage │  │ Key Vault   │  │ Azure Functions │  │  │
│  │  │ (Data Lake)  │  │ (Secrets)   │  │ (xlsx→parquet)  │  │  │
│  │  └──────┬───────┘  └──────┬──────┘  └────────┬────────┘  │  │
│  │         │                   │                  │            │  │
│  │  ┌──────┴───────────────────┴──────────────────┴─────────┐  │
│  │  │                                                         │  │
│  │  │            Nomad Cluster (1-3 VMs)                    │  │
│  │  │                                                         │  │
│  │  │  ┌────────────────────┐    ┌──────────────────────┐  │  │
│  │  │  │   ML Pipeline      │    │     Dashboard        │  │  │
│  │  │  │   (Batch Job)      │    │     (Service)        │  │  │
│  │  │  │                    │    │                      │  │  │
│  │  │  │ • Ensemble Models  │    │ • Dash Web App      │  │  │
│  │  │  │ • SNA Analysis     │    │ • Plotly Charts     │  │  │
│  │  │  │ • Risk Scoring     │    │ • Network Viz       │  │  │
│  │  │  └────────────────────┘    └──────────────────────┘  │  │
│  │  │                                                         │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                            │  │
│  │  ┌──────────────┐                                         │  │
│  │  │Load Balancer │  → Public Access                        │  │
│  │  └──────────────┘                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              CI/CD Pipeline (GitHub Actions)              │  │
│  │  • Build Docker images                                    │  │
│  │  • Run tests                                              │  │
│  │  • Deploy to Azure & Nomad                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Data Storage Layer

#### Azure Blob Storage
**Purpose**: Centralized data lake for all application data

**Containers**:
- `raw-data`: Raw Excel files uploaded by users
- `processed-data`: Parquet files (base1, base2)
- `model-outputs`: ML model results and reports

**Characteristics**:
- Hot tier for frequent access
- 7-day soft delete retention
- Versioning enabled
- LRS replication (dev), GRS (prod)

**Cost**: ~$2-5/month (dev)

---

### 2. Compute Layer

#### Azure Functions (Serverless)
**Purpose**: Lightweight data transformation (xlsx → parquet)

**Trigger**: Blob upload to `raw-data` container

**Execution**:
1. Receives blob upload event
2. Reads Excel file (both sheets)
3. Applies transformations
4. Writes parquet files to `processed-data`
5. ~1-2 minutes per file

**Resources**:
- Consumption plan (Y1 SKU)
- Python 3.11 runtime
- 512MB-1GB memory
- 10 min timeout

**Cost**: ~$0-1/month (within free tier)

---

#### Nomad Cluster
**Purpose**: Orchestrate ML workloads and dashboard service

**VMs**:
- **Dev**: 1x Standard_B2ms (2 vCPU, 8GB RAM)
- **Prod**: 3x Standard_B2ms (HA cluster)

**Architecture**:
- Node 0: Server + Client (primary)
- Node 1-2: Clients only (if prod)

**Networking**:
- Virtual Network: 10.0.0.0/16
- Subnet: 10.0.1.0/24
- Public IPs for each VM
- NSG: SSH (22), Nomad (4646), Dashboard (8060)

**Cost**: ~$30-90/month depending on VM count

---

### 3. Application Layer

#### ML Pipeline (Nomad Batch Job)
**Purpose**: Run machine learning models and analysis

**Schedule**: Daily at 2 AM (America/Sao_Paulo)

**Workflow**:
```
1. Download parquet files from Blob Storage
2. Build feature engineering (f_empresa_mes)
3. Run classification models:
   - Simple rule-based
   - Unsupervised clustering
   - Supervised ML (Random Forest, Gradient Boosting)
   - Ensemble model
4. Social network analysis (SNA)
5. Sector benchmarking
6. Risk scoring
7. Upload results to Blob Storage
```

**Resources**:
- 4 CPU cores
- 8GB RAM
- ~30-60 min execution time

**Docker Image**: `ghcr.io/{repo}/ml-pipeline:latest`

**Key Technologies**:
- pandas, numpy, scipy
- scikit-learn
- networkx (SNA)
- duckdb (queries)

---

#### Dashboard (Nomad Service)
**Purpose**: Interactive web application for data exploration

**Features**:
- **Desafio 1**: Company profiles, stage classification
- **Desafio 2**: Relationship networks, risk analysis

**Technologies**:
- Dash (Plotly)
- Dash Bootstrap Components
- Dash Cytoscape (network viz)

**Resources**:
- 1 CPU core
- 2GB RAM
- Port 8060

**Data Sync**:
- Reads from Blob Storage every 5 minutes
- Displays latest model outputs

**Docker Image**: `ghcr.io/{repo}/dashboard:latest`

**Access**:
- Via Load Balancer: `http://<lb-public-ip>`
- Direct: `http://<vm-ip>:8060`

---

### 4. Security Layer

#### Azure Key Vault
**Purpose**: Centralized secrets management

**Stored Secrets**:
- Storage account connection strings
- Azure service credentials
- Nomad ACL tokens (if enabled)
- Database passwords (future)

**Access**:
- Azure Functions: System-assigned managed identity
- VMs: DefaultAzureCredential
- CI/CD: Service principal

**Cost**: ~$0.03/10K operations (negligible)

---

### 5. CI/CD Pipeline

#### GitHub Actions Workflows

##### 1. `ci-cd.yml` (Main Pipeline)
**Triggers**: Push to main/develop, Pull Requests

**Jobs**:
```
┌──────────┐
│   Test   │ → Lint (ruff) + pytest + coverage
└────┬─────┘
     ↓
┌──────────────┐
│ Build Images │ → Build ML Pipeline + Dashboard containers
└──────┬───────┘
       ↓
┌────────────────────┐
│ Deploy Function    │ → Deploy Azure Function
└──────┬─────────────┘
       ↓
┌────────────────────┐
│ Deploy Nomad Jobs  │ → Update Nomad cluster
└────────────────────┘
```

**Artifacts**:
- Docker images → GitHub Container Registry
- Function package → Azure Functions
- Updated Nomad job specs

##### 2. `terraform-apply.yml` (Infrastructure)
**Trigger**: Manual workflow dispatch

**Actions**:
- `plan`: Preview infrastructure changes
- `apply`: Create/update resources
- `destroy`: Cleanup resources

---

## Data Flow

### ETL Pipeline

```
┌────────────────┐
│  Upload .xlsx  │
│  to Blob       │
└───────┬────────┘
        ↓
┌───────────────────────┐
│ Azure Function        │
│ • Read Excel          │
│ • Transform           │
│ • Write Parquet       │
└───────┬───────────────┘
        ↓
┌───────────────────────┐
│ Blob Storage          │
│ processed-data/       │
│ ├── base1/            │
│ └── base2/            │
└───────┬───────────────┘
        ↓
┌───────────────────────┐
│ Nomad ML Pipeline     │
│ (Daily 2 AM)          │
│ • Download data       │
│ • Run models          │
│ • Generate reports    │
└───────┬───────────────┘
        ↓
┌───────────────────────┐
│ Blob Storage          │
│ model-outputs/        │
│ ├── estagio.parquet   │
│ ├── relations.parquet │
│ └── ...               │
└───────┬───────────────┘
        ↓
┌───────────────────────┐
│ Dashboard             │
│ • Sync from blob      │
│ • Display results     │
└───────────────────────┘
```

### User Interaction Flow

```
User → Load Balancer → Dashboard Service → Blob Storage
                                         ↓
                                    Render Charts
                                         ↓
                                    Return HTML
```

---

## Scalability

### Horizontal Scaling

**Dashboard**:
```hcl
# In nomad/dashboard.nomad
group "web" {
  count = 3  # Scale to 3 instances
  ...
}
```

**Load Balancer**: Automatically distributes traffic

### Vertical Scaling

**VMs**:
```hcl
# In terraform/variables.tf
nomad_vm_size = "Standard_B4ms"  # Upgrade to 4 vCPU, 16GB RAM
```

### Auto-Scaling (Future)

- Azure VM Scale Sets
- Nomad autoscaler plugin
- Scale based on:
  - CPU utilization
  - Memory usage
  - Queue depth

---

## High Availability

### Current Setup (Dev)
- 1 VM: No HA
- Acceptable for development

### Production Setup
```
3 Nomad VMs:
├── VM-0: Server + Client (Leader)
├── VM-1: Client (Standby)
└── VM-2: Client (Standby)

Load Balancer:
├── Health checks every 10s
├── Automatic failover
└── Session affinity (optional)
```

**RPO/RTO**:
- Recovery Point Objective: < 1 day (daily backups)
- Recovery Time Objective: < 5 minutes (automatic failover)

---

## Security

### Network Security

```
Internet
    ↓
[NSG - Firewall Rules]
    ├── Allow: SSH (22) from anywhere
    ├── Allow: HTTP (80) from anywhere
    ├── Allow: Nomad UI (4646) from anywhere
    └── Allow: Dashboard (8060) from anywhere
    ↓
[Virtual Network]
    ↓
[VMs - Private Network]
```

**Improvements** (production):
- Restrict SSH to VPN/bastion
- Use HTTPS with SSL certificates
- Enable Azure DDoS Protection
- Implement Web Application Firewall (WAF)

### Identity & Access

**Authentication**:
- VMs: SSH key-based (no passwords)
- Azure: Service principal + managed identities
- Blob Storage: Connection strings in Key Vault

**Authorization**:
- RBAC for Azure resources
- Nomad ACL (optional, not enabled by default)

### Data Security

- **Encryption at rest**: Azure Storage Service Encryption (SSE)
- **Encryption in transit**: HTTPS/TLS for all API calls
- **Secrets**: Stored in Azure Key Vault, never in code

---

## Monitoring & Logging

### Current Logging

**Azure Functions**:
```bash
func azure functionapp logstream $FUNC_APP_NAME
```

**Nomad Jobs**:
```bash
nomad alloc logs <alloc-id>
```

**VMs**:
```bash
sudo journalctl -u nomad -f
```

### Recommended Monitoring (Future)

**Azure Monitor**:
- VM metrics (CPU, memory, disk)
- Storage metrics (requests, latency)
- Function metrics (executions, errors)

**Application Insights**:
- Application performance monitoring (APM)
- Request tracing
- Exception tracking

**Grafana + Prometheus**:
- Custom dashboards
- Nomad cluster metrics
- ML pipeline performance

---

## Disaster Recovery

### Backup Strategy

**Data**:
- Blob Storage: Versioning enabled (7 days)
- Manual backups: `az storage blob download-batch`

**Infrastructure**:
- Terraform state: Stored in Azure (optional)
- Code: GitHub repository

### Recovery Procedures

**Scenario 1: VM failure**
```bash
# Terraform will recreate from state
terraform apply -var-file=environments/prod.tfvars
```

**Scenario 2: Data corruption**
```bash
# Restore from blob version
az storage blob restore --account-name $STORAGE --source VERSION_ID
```

**Scenario 3: Complete disaster**
```bash
# 1. Restore code from GitHub
# 2. Re-run Terraform
# 3. Restore data from backups
# Time: ~30-60 minutes
```

---

## Performance Optimization

### Current Performance

| Component | Metric | Value |
|-----------|--------|-------|
| Function | Execution time | ~1-2 min |
| ML Pipeline | Total runtime | ~30-60 min |
| Dashboard | Page load | ~2-3 sec |
| API Response | P95 latency | <500ms |

### Optimization Opportunities

1. **ML Pipeline**:
   - Cache intermediate results
   - Parallelize model training
   - Use GPU instances for deep learning

2. **Dashboard**:
   - Implement Redis cache
   - Pre-aggregate data
   - Use CDN for static assets

3. **Storage**:
   - Use Azure CDN for hot data
   - Move old data to Cool tier
   - Implement data lifecycle policies

---

## Cost Optimization

### Current Costs (Dev Environment)

| Component | Monthly Cost | Notes |
|-----------|--------------|-------|
| Blob Storage | $2-3 | Hot tier, <10GB |
| Azure Functions | $0-1 | Within free tier |
| Key Vault | <$1 | Low operation count |
| VM (1x B2ms) | $30-35 | 730 hrs/month |
| Load Balancer | $20-25 | Standard SKU |
| Networking | $2-5 | Bandwidth |
| **Total** | **~$55-70** | |

### Cost Savings

1. **Auto-shutdown**: Save ~60% on dev VMs
2. **Spot VMs**: Save 60-90% on compute
3. **Reserved Instances**: Save 40-60% (1-year commitment)
4. **Cool Storage Tier**: Save ~50% on old data

**With Optimizations**: ~$30-40/month for dev

---

## Future Enhancements

### Short Term (1-3 months)
- [ ] HTTPS/SSL certificates
- [ ] Azure Monitor integration
- [ ] Automated backups
- [ ] User authentication

### Medium Term (3-6 months)
- [ ] Multi-region deployment
- [ ] Advanced monitoring (Grafana)
- [ ] API endpoints (FastAPI)
- [ ] Real-time data streaming

### Long Term (6-12 months)
- [ ] Kubernetes migration
- [ ] MLOps pipeline (MLflow)
- [ ] AutoML capabilities
- [ ] Mobile application

---

## References

- [Azure Architecture Center](https://docs.microsoft.com/azure/architecture/)
- [Nomad Documentation](https://www.nomadproject.io/docs)
- [Terraform Azure Provider](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Dash Documentation](https://dash.plotly.com/)


