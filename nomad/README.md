# Nomad Job Definitions

This directory contains Nomad job specifications for the FIAP Challenge application.

## Jobs

### 1. ML Pipeline (`ml-pipeline.nomad`)
- **Type**: Batch (Periodic)
- **Schedule**: Daily at 2 AM (America/Sao_Paulo)
- **Resources**: 4 CPU cores, 8GB RAM
- **Purpose**: 
  - Download processed data from Azure Blob
  - Run ML models and analysis
  - Upload results back to Azure Blob

### 2. Dashboard (`dashboard.nomad`)
- **Type**: Service (Long-running)
- **Port**: 8060
- **Resources**: 1 CPU core, 2GB RAM
- **Purpose**: 
  - Serve Dash web application
  - Automatically sync reports from Azure Blob every 5 minutes

## Prerequisites

Before deploying these jobs, you need to:

1. **Setup Nomad Key-Value Store** with secrets:
   ```bash
   nomad var put azure/storage_account "your-storage-account-name"
   nomad var put azure/storage_key "your-storage-account-key"
   nomad var put azure/key_vault_uri "https://your-keyvault.vault.azure.net/"
   nomad var put docker/username "your-github-username"
   nomad var put docker/password "your-github-token"
   ```

2. **Update Docker image URLs** in both job files:
   - Replace `YOUR_GITHUB_USERNAME` with your GitHub username
   - Replace `REPO_NAME` with your repository name

3. **Ensure Docker images are pushed** to GitHub Container Registry

## Deployment

### Deploy ML Pipeline
```bash
nomad job run nomad/ml-pipeline.nomad
```

### Deploy Dashboard
```bash
nomad job run nomad/dashboard.nomad
```

### Check Status
```bash
# Check job status
nomad job status ml-pipeline
nomad job status dashboard

# View logs
nomad alloc logs <alloc-id>

# Follow logs
nomad alloc logs -f <alloc-id>
```

### Trigger Pipeline Manually
```bash
nomad job dispatch ml-pipeline
```

### Scale Dashboard
```bash
# Edit count in dashboard.nomad, then:
nomad job run nomad/dashboard.nomad
```

## Monitoring

### Dashboard Health
The dashboard exposes a health check at `http://<nomad-vm-ip>:8060/`

### Pipeline Execution
Check pipeline execution logs:
```bash
nomad job status ml-pipeline
nomad alloc logs $(nomad job status ml-pipeline -json | jq -r '.ID')
```

## Troubleshooting

### Pipeline Fails
1. Check logs: `nomad alloc logs <alloc-id>`
2. Verify Azure credentials in Nomad variables
3. Ensure blob storage has processed data
4. Check resource allocation (may need more memory)

### Dashboard Not Accessible
1. Check service health: `nomad job status dashboard`
2. Verify port 8060 is open in NSG
3. Check load balancer configuration
4. View logs: `nomad alloc logs <alloc-id>`

### Images Not Pulling
1. Verify GitHub Container Registry access
2. Check Docker credentials in Nomad variables
3. Try pulling manually: `docker pull ghcr.io/...`

## Resource Optimization

For **development** environment:
- ML Pipeline: 2 CPU, 4GB RAM
- Dashboard: 500m CPU, 1GB RAM

For **production** environment:
- ML Pipeline: 4-8 CPU, 8-16GB RAM
- Dashboard: 1-2 CPU, 2-4GB RAM (scale to 2-3 instances)


