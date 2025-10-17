# Development Environment Configuration
project_name = "fiapchallenge"
environment  = "dev"
location     = "brazilsouth"

# VM Configuration - Minimal for dev
nomad_vm_count = 1
nomad_vm_size  = "Standard_B2ms"  # 2 vCPU, 8GB RAM

# Admin Access (set via command line or CI/CD)
admin_username = "azureuser"
# admin_ssh_public_key = "set via -var flag"

# Auto Shutdown
enable_auto_shutdown = true
auto_shutdown_time   = "1900"  # 7 PM BRT


