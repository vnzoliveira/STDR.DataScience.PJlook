# Nomad Cluster Module
resource "azurerm_public_ip" "nomad_vm" {
  count               = var.vm_count
  name                = "${var.project_name}-${var.environment}-nomad-vm-${count.index}-ip"
  location            = var.location
  resource_group_name = var.resource_group_name
  allocation_method   = "Static"
  sku                 = "Standard"
  
  tags = var.tags
}

resource "azurerm_network_interface" "nomad_vm" {
  count               = var.vm_count
  name                = "${var.project_name}-${var.environment}-nomad-vm-${count.index}-nic"
  location            = var.location
  resource_group_name = var.resource_group_name
  
  ip_configuration {
    name                          = "internal"
    subnet_id                     = var.subnet_id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.nomad_vm[count.index].id
  }
  
  tags = var.tags
}

resource "azurerm_network_interface_security_group_association" "nomad_vm" {
  count                     = var.vm_count
  network_interface_id      = azurerm_network_interface.nomad_vm[count.index].id
  network_security_group_id = var.nsg_id
}

resource "azurerm_linux_virtual_machine" "nomad_vm" {
  count               = var.vm_count
  name                = "${var.project_name}-${var.environment}-nomad-vm-${count.index}"
  resource_group_name = var.resource_group_name
  location            = var.location
  size                = var.vm_size
  admin_username      = var.admin_username
  
  network_interface_ids = [
    azurerm_network_interface.nomad_vm[count.index].id,
  ]
  
  admin_ssh_key {
    username   = var.admin_username
    public_key = var.admin_ssh_key
  }
  
  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 30
  }
  
  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }
  
  # Bootstrap script
  custom_data = base64encode(templatefile("${path.module}/scripts/init-nomad.sh", {
    is_server  = count.index == 0 ? "true" : "false"
    server_ip  = count.index == 0 ? azurerm_network_interface.nomad_vm[0].private_ip_address : azurerm_network_interface.nomad_vm[0].private_ip_address
    node_index = count.index
  }))
  
  identity {
    type = "SystemAssigned"
  }
  
  tags = merge(var.tags, {
    Role = count.index == 0 ? "nomad-server" : "nomad-client"
  })
}

# Auto-shutdown schedule for dev environment
resource "azurerm_dev_test_global_vm_shutdown_schedule" "nomad_vm" {
  count              = var.environment == "dev" ? var.vm_count : 0
  virtual_machine_id = azurerm_linux_virtual_machine.nomad_vm[count.index].id
  location           = var.location
  enabled            = true
  
  daily_recurrence_time = "1900"
  timezone              = "E. South America Standard Time"
  
  notification_settings {
    enabled = false
  }
}


