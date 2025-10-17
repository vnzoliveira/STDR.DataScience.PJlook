# Production Environment Configuration
project_name = "fiapchallenge"
environment  = "prod"
location     = "brazilsouth"

# VM Configuration - High availability
nomad_vm_count = 3
nomad_vm_size  = "Standard_B2ms"  # 2 vCPU, 8GB RAM

# Admin Access (set via command line or CI/CD)
admin_username = "azureuser"
# admin_ssh_public_key = "set via -var flag"

# Auto Shutdown - Disabled for production
enable_auto_shutdown = false


