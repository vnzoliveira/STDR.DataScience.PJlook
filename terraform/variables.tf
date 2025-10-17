# Terraform Variables
variable "project_name" {
  description = "Project name (used as prefix for resources)"
  type        = string
  default     = "fiapchallenge"
  
  validation {
    condition     = can(regex("^[a-z0-9]{3,15}$", var.project_name))
    error_message = "Project name must be lowercase alphanumeric, 3-15 characters."
  }
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "brazilsouth"
}

variable "nomad_vm_count" {
  description = "Number of Nomad VMs (1 for dev, 3 for prod)"
  type        = number
  default     = 1
  
  validation {
    condition     = var.nomad_vm_count >= 1 && var.nomad_vm_count <= 5
    error_message = "VM count must be between 1 and 5."
  }
}

variable "nomad_vm_size" {
  description = "VM size for Nomad nodes"
  type        = string
  default     = "Standard_B2ms"
}

variable "admin_username" {
  description = "Admin username for VMs"
  type        = string
  default     = "azureuser"
}

variable "admin_ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
}

variable "enable_auto_shutdown" {
  description = "Enable automatic VM shutdown (dev only)"
  type        = bool
  default     = true
}

variable "auto_shutdown_time" {
  description = "Auto shutdown time (24h format, e.g., 1900 for 7 PM)"
  type        = string
  default     = "1900"
}


