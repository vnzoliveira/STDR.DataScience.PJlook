# Nomad Cluster Module Outputs
output "vm_ids" {
  description = "IDs of the VMs"
  value       = azurerm_linux_virtual_machine.nomad_vm[*].id
}

output "public_ips" {
  description = "Public IPs of the VMs"
  value       = azurerm_public_ip.nomad_vm[*].ip_address
}

output "private_ips" {
  description = "Private IPs of the VMs"
  value       = azurerm_network_interface.nomad_vm[*].private_ip_address
}

output "vm_names" {
  description = "Names of the VMs"
  value       = azurerm_linux_virtual_machine.nomad_vm[*].name
}


