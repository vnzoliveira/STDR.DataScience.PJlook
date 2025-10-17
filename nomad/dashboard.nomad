# Nomad Job: Dashboard (Service)
job "dashboard" {
  datacenters = ["dc1"]
  type        = "service"
  
  group "web" {
    count = 1  # Scale to 2-3 for HA in production
    
    network {
      port "http" {
        to = 8060
      }
    }
    
    restart {
      attempts = 3
      delay    = "15s"
      interval = "2m"
      mode     = "fail"
    }
    
    update {
      max_parallel     = 1
      min_healthy_time = "30s"
      healthy_deadline = "3m"
      auto_revert      = true
    }
    
    task "dash-app" {
      driver = "docker"
      
      config {
        image = "ghcr.io/YOUR_GITHUB_USERNAME/REPO_NAME/dashboard:latest"
        
        auth {
          username = "${DOCKER_USERNAME}"
          password = "${DOCKER_PASSWORD}"
        }
        
        ports = ["http"]
        
        volumes = [
          "/opt/nomad/data/dashboard/reports:/app/reports:ro"
        ]
        
        # Periodically sync reports from blob storage
        command = "/bin/bash"
        args = [
          "-c",
          <<EOF
# Background sync job
(
  while true; do
    echo "Syncing reports from Azure Blob..."
    az storage blob download-batch \
      --source model-outputs \
      --destination /app/reports/exports \
      --account-name ${AZURE_STORAGE_ACCOUNT} \
      --account-key ${AZURE_STORAGE_KEY} \
      --overwrite
    sleep 300  # Sync every 5 minutes
  done
) &

# Start dashboard
python -m src.ui.app
EOF
        ]
      }
      
      resources {
        cpu    = 1000  # 1 core
        memory = 2048  # 2 GB
      }
      
      # Environment variables
      template {
        data = <<EOH
AZURE_STORAGE_ACCOUNT="{{ key "azure/storage_account" }}"
AZURE_STORAGE_KEY="{{ key "azure/storage_key" }}"
KEY_VAULT_URI="{{ key "azure/key_vault_uri" }}"
DOCKER_USERNAME="{{ key "docker/username" }}"
DOCKER_PASSWORD="{{ key "docker/password" }}"
PORT="8060"
EOH
        destination = "secrets/env.txt"
        env         = true
      }
      
      # Health check
      service {
        name = "dashboard"
        port = "http"
        
        tags = [
          "fiap",
          "dashboard",
          "dash"
        ]
        
        check {
          type     = "http"
          path     = "/"
          interval = "10s"
          timeout  = "2s"
        }
      }
      
      logs {
        max_files     = 5
        max_file_size = 10
      }
    }
  }
}


