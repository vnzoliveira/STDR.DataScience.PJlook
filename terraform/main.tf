# Main Terraform configuration for FIAP Challenge Azure Infrastructure
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  
  # Store state in Azure (uncomment after initial setup)
  # backend "azurerm" {
  #   resource_group_name  = "fiap-terraform-state"
  #   storage_account_name = "fiaptfstate"
  #   container_name       = "tfstate"
  #   key                  = "fiap-challenge.tfstate"
  # }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
  }
}

data "azurerm_client_config" "current" {}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-${var.environment}"
  location = var.location
  
  tags = {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Team        = "FIAP-DataScience"
  }
}

# Storage Account for data lake
resource "azurerm_storage_account" "datalake" {
  name                     = "${var.project_name}${var.environment}sa"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.environment == "prod" ? "GRS" : "LRS"
  
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 7
    }
  }
  
  tags = azurerm_resource_group.main.tags
}

# Blob Containers
resource "azurerm_storage_container" "raw_data" {
  name                  = "raw-data"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "processed_data" {
  name                  = "processed-data"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "model_outputs" {
  name                  = "model-outputs"
  storage_account_name  = azurerm_storage_account.datalake.name
  container_access_type = "private"
}

# Key Vault for secrets
resource "azurerm_key_vault" "main" {
  name                = "${var.project_name}-${var.environment}-kv"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  
  purge_protection_enabled   = false
  soft_delete_retention_days = 7
  
  tags = azurerm_resource_group.main.tags
}

# Store storage connection string in Key Vault
resource "azurerm_key_vault_secret" "storage_connection" {
  name         = "storage-connection-string"
  value        = azurerm_storage_account.datalake.primary_connection_string
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.terraform]
}

# Key Vault access policy for Terraform
resource "azurerm_key_vault_access_policy" "terraform" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id
  
  secret_permissions = [
    "Get", "List", "Set", "Delete", "Purge", "Recover"
  ]
}

# App Service Plan for Functions
resource "azurerm_service_plan" "function_plan" {
  name                = "${var.project_name}-${var.environment}-func-plan"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "Y1" # Consumption plan - FREE
  
  tags = azurerm_resource_group.main.tags
}

# Storage Account for Functions
resource "azurerm_storage_account" "function_storage" {
  name                     = "${var.project_name}${var.environment}funcsa"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  tags = azurerm_resource_group.main.tags
}

# Azure Function App
resource "azurerm_linux_function_app" "xlsx_parser" {
  name                = "${var.project_name}-${var.environment}-func"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  
  storage_account_name       = azurerm_storage_account.function_storage.name
  storage_account_access_key = azurerm_storage_account.function_storage.primary_access_key
  service_plan_id            = azurerm_service_plan.function_plan.id
  
  site_config {
    application_stack {
      python_version = "3.11"
    }
  }
  
  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME"       = "python"
    "AzureWebJobsStorage"            = azurerm_storage_account.function_storage.primary_connection_string
    "STORAGE_CONNECTION_STRING"      = azurerm_storage_account.datalake.primary_connection_string
    "KEY_VAULT_URI"                  = azurerm_key_vault.main.vault_uri
    "ENABLE_ORYX_BUILD"              = "true"
    "SCM_DO_BUILD_DURING_DEPLOYMENT" = "true"
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = azurerm_resource_group.main.tags
}

# Grant Function access to Key Vault
resource "azurerm_key_vault_access_policy" "function_app" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_linux_function_app.xlsx_parser.identity[0].principal_id
  
  secret_permissions = ["Get", "List"]
}

# Virtual Network for Nomad cluster
resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-${var.environment}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  tags = azurerm_resource_group.main.tags
}

resource "azurerm_subnet" "nomad" {
  name                 = "nomad-subnet"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Network Security Group
resource "azurerm_network_security_group" "nomad" {
  name                = "${var.project_name}-${var.environment}-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  # SSH
  security_rule {
    name                       = "SSH"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  # Nomad HTTP API
  security_rule {
    name                       = "Nomad-HTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "4646"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  # Dashboard
  security_rule {
    name                       = "Dashboard"
    priority                   = 1003
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8060"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
  
  # Nomad Serf (cluster communication)
  security_rule {
    name                       = "Nomad-Serf"
    priority                   = 1004
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "4648"
    source_address_prefix      = "10.0.0.0/16"
    destination_address_prefix = "*"
  }
  
  tags = azurerm_resource_group.main.tags
}

# Nomad VMs
module "nomad_cluster" {
  source = "./modules/nomad-cluster"
  
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  subnet_id           = azurerm_subnet.nomad.id
  nsg_id              = azurerm_network_security_group.nomad.id
  
  vm_count            = var.nomad_vm_count
  vm_size             = var.nomad_vm_size
  admin_username      = var.admin_username
  admin_ssh_key       = var.admin_ssh_public_key
  
  environment         = var.environment
  project_name        = var.project_name
  
  tags                = azurerm_resource_group.main.tags
}

# Public IP for Load Balancer
resource "azurerm_public_ip" "lb" {
  name                = "${var.project_name}-${var.environment}-lb-ip"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  allocation_method   = "Static"
  sku                 = "Standard"
  
  tags = azurerm_resource_group.main.tags
}

# Load Balancer for Dashboard
resource "azurerm_lb" "main" {
  name                = "${var.project_name}-${var.environment}-lb"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Standard"
  
  frontend_ip_configuration {
    name                 = "PublicIPAddress"
    public_ip_address_id = azurerm_public_ip.lb.id
  }
  
  tags = azurerm_resource_group.main.tags
}

# Backend Pool
resource "azurerm_lb_backend_address_pool" "main" {
  loadbalancer_id = azurerm_lb.main.id
  name            = "nomad-backend-pool"
}

# Health Probe
resource "azurerm_lb_probe" "dashboard" {
  loadbalancer_id = azurerm_lb.main.id
  name            = "dashboard-health-probe"
  port            = 8060
  protocol        = "Http"
  request_path    = "/"
}

# Load Balancer Rule
resource "azurerm_lb_rule" "dashboard" {
  loadbalancer_id                = azurerm_lb.main.id
  name                           = "dashboard-rule"
  protocol                       = "Tcp"
  frontend_port                  = 80
  backend_port                   = 8060
  frontend_ip_configuration_name = "PublicIPAddress"
  backend_address_pool_ids       = [azurerm_lb_backend_address_pool.main.id]
  probe_id                       = azurerm_lb_probe.dashboard.id
}


