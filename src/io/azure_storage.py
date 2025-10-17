"""
Azure Blob Storage integration utilities
"""
import os
import logging
from pathlib import Path
from typing import Optional, List
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)


class AzureBlobStorage:
    """Azure Blob Storage client wrapper"""
    
    def __init__(self, connection_string: Optional[str] = None, account_name: Optional[str] = None):
        """
        Initialize Azure Blob Storage client
        
        Args:
            connection_string: Azure Storage connection string (from env or Key Vault)
            account_name: Storage account name (if using DefaultAzureCredential)
        """
        if connection_string:
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        elif account_name:
            account_url = f"https://{account_name}.blob.core.windows.net"
            credential = DefaultAzureCredential()
            self.blob_service_client = BlobServiceClient(account_url, credential=credential)
        else:
            # Try to get from environment
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            else:
                raise ValueError("Must provide connection_string, account_name, or set AZURE_STORAGE_CONNECTION_STRING env var")
    
    def download_blob(self, container_name: str, blob_name: str, local_path: str) -> bool:
        """
        Download a blob to local file
        
        Args:
            container_name: Container name
            blob_name: Blob name (path)
            local_path: Local file path to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Ensure directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, "wb") as f:
                blob_data = blob_client.download_blob()
                f.write(blob_data.readall())
            
            logger.info(f"Downloaded {blob_name} to {local_path}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"Blob not found: {container_name}/{blob_name}")
            return False
        except Exception as e:
            logger.error(f"Error downloading blob: {e}")
            return False
    
    def upload_blob(self, container_name: str, blob_name: str, local_path: str, overwrite: bool = True) -> bool:
        """
        Upload a local file to blob storage
        
        Args:
            container_name: Container name
            blob_name: Blob name (path)
            local_path: Local file path to upload
            overwrite: Whether to overwrite existing blob
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=overwrite)
            
            logger.info(f"Uploaded {local_path} to {blob_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading blob: {e}")
            return False
    
    def download_directory(self, container_name: str, blob_prefix: str, local_directory: str) -> int:
        """
        Download all blobs with a prefix to local directory
        
        Args:
            container_name: Container name
            blob_prefix: Blob prefix (directory path)
            local_directory: Local directory to save files
            
        Returns:
            Number of files downloaded
        """
        count = 0
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = container_client.list_blobs(name_starts_with=blob_prefix)
            
            for blob in blobs:
                # Calculate local path
                relative_path = blob.name[len(blob_prefix):].lstrip('/')
                local_path = os.path.join(local_directory, relative_path)
                
                if self.download_blob(container_name, blob.name, local_path):
                    count += 1
            
            logger.info(f"Downloaded {count} files from {container_name}/{blob_prefix}")
            return count
            
        except Exception as e:
            logger.error(f"Error downloading directory: {e}")
            return count
    
    def upload_directory(self, container_name: str, local_directory: str, blob_prefix: str = "") -> int:
        """
        Upload all files from local directory to blob storage
        
        Args:
            container_name: Container name
            local_directory: Local directory to upload
            blob_prefix: Blob prefix (directory path)
            
        Returns:
            Number of files uploaded
        """
        count = 0
        try:
            local_path = Path(local_directory)
            
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Calculate blob name
                    relative_path = file_path.relative_to(local_path)
                    blob_name = os.path.join(blob_prefix, str(relative_path)).replace('\\', '/')
                    
                    if self.upload_blob(container_name, blob_name, str(file_path)):
                        count += 1
            
            logger.info(f"Uploaded {count} files to {container_name}/{blob_prefix}")
            return count
            
        except Exception as e:
            logger.error(f"Error uploading directory: {e}")
            return count
    
    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> List[str]:
        """
        List all blobs in a container with optional prefix
        
        Args:
            container_name: Container name
            prefix: Optional blob prefix filter
            
        Returns:
            List of blob names
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            if prefix:
                blobs = container_client.list_blobs(name_starts_with=prefix)
            else:
                blobs = container_client.list_blobs()
            
            return [blob.name for blob in blobs]
            
        except Exception as e:
            logger.error(f"Error listing blobs: {e}")
            return []


def get_blob_client() -> AzureBlobStorage:
    """
    Get configured Azure Blob Storage client
    
    Returns:
        AzureBlobStorage instance
    """
    # Try connection string first
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("STORAGE_CONNECTION_STRING")
    
    if connection_string:
        return AzureBlobStorage(connection_string=connection_string)
    
    # Try account name with managed identity
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    if account_name:
        return AzureBlobStorage(account_name=account_name)
    
    raise ValueError("Azure Storage credentials not configured")


