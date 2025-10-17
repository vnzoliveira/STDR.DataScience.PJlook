# Terraform Outputs
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.datalake.name
}

output "storage_connection_string" {
  description = "Storage account connection string"
  value       = azurerm_storage_account.datalake.primary_connection_string
  sensitive   = true
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = azurerm_key_vault.main.vault_uri
}

output "function_app_name" {
  description = "Name of the Azure Function App"
  value       = azurerm_linux_function_app.xlsx_parser.name
}

output "function_app_url" {
  description = "URL of the Azure Function App"
  value       = "https://${azurerm_linux_function_app.xlsx_parser.default_hostname}"
}

output "nomad_vm_public_ips" {
  description = "Public IPs of Nomad VMs"
  value       = module.nomad_cluster.public_ips
}

output "nomad_vm_private_ips" {
  description = "Private IPs of Nomad VMs"
  value       = module.nomad_cluster.private_ips
}

output "dashboard_url" {
  description = "URL of the dashboard (via load balancer)"
  value       = "http://${azurerm_public_ip.lb.ip_address}"
}

output "nomad_ui_url" {
  description = "URL of the Nomad UI"
  value       = "http://${module.nomad_cluster.public_ips[0]}:4646"
}

output "ssh_commands" {
  description = "SSH commands to connect to Nomad VMs"
  value       = [for i, ip in module.nomad_cluster.public_ips : "ssh ${var.admin_username}@${ip}"]
}


