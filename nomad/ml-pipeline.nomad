# Nomad Job: ML Pipeline (Batch Job)
job "ml-pipeline" {
  datacenters = ["dc1"]
  type        = "batch"
  
  # Run daily at 2 AM
  periodic {
    cron             = "0 2 * * *"
    prohibit_overlap = true
    time_zone        = "America/Sao_Paulo"
  }
  
  group "pipeline" {
    count = 1
    
    restart {
      attempts = 2
      delay    = "15s"
      interval = "30m"
      mode     = "fail"
    }
    
    task "build-all" {
      driver = "docker"
      
      config {
        image = "ghcr.io/YOUR_GITHUB_USERNAME/REPO_NAME/ml-pipeline:latest"
        
        # Azure CLI needs to authenticate
        auth {
          username = "${DOCKER_USERNAME}"
          password = "${DOCKER_PASSWORD}"
        }
        
        volumes = [
          "/opt/nomad/data/ml-pipeline:/app/data",
          "/opt/nomad/data/ml-pipeline/reports:/app/reports"
        ]
        
        # Use Azure CLI to download/upload data
        command = "/bin/bash"
        args = [
          "-c",
          <<EOF
set -e
echo "Downloading data from Azure Blob..."
az storage blob download-batch \
  --source processed-data \
  --destination /app/data/processed \
  --pattern "*.parquet" \
  --account-name ${AZURE_STORAGE_ACCOUNT} \
  --account-key ${AZURE_STORAGE_KEY}

echo "Running ML pipeline..."
python -m src.cli.main build-all

echo "Uploading results to Azure Blob..."
az storage blob upload-batch \
  --source /app/reports/exports \
  --destination model-outputs \
  --account-name ${AZURE_STORAGE_ACCOUNT} \
  --account-key ${AZURE_STORAGE_KEY}

echo "Pipeline completed successfully!"
EOF
        ]
      }
      
      # Allocate resources for ML workload
      resources {
        cpu    = 4000  # 4 cores
        memory = 8192  # 8 GB
      }
      
      # Environment variables from Nomad variables or Vault
      template {
        data = <<EOH
AZURE_STORAGE_ACCOUNT="{{ key "azure/storage_account" }}"
AZURE_STORAGE_KEY="{{ key "azure/storage_key" }}"
KEY_VAULT_URI="{{ key "azure/key_vault_uri" }}"
DOCKER_USERNAME="{{ key "docker/username" }}"
DOCKER_PASSWORD="{{ key "docker/password" }}"
EOH
        destination = "secrets/env.txt"
        env         = true
      }
      
      # Logs
      logs {
        max_files     = 5
        max_file_size = 10
      }
    }
  }
}


