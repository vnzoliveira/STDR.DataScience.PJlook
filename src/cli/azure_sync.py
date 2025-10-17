"""
Azure Blob Storage sync utilities for CLI
"""
import argparse
import logging
from pathlib import Path
from ..io.azure_storage import get_blob_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def download_processed_data():
    """Download processed parquet files from Azure Blob"""
    logger.info("Downloading processed data from Azure Blob...")
    
    blob_client = get_blob_client()
    
    # Download base1
    count_base1 = blob_client.download_directory(
        container_name="processed-data",
        blob_prefix="base1/",
        local_directory="data/processed/base1"
    )
    
    # Download base2
    count_base2 = blob_client.download_directory(
        container_name="processed-data",
        blob_prefix="base2/",
        local_directory="data/processed/base2"
    )
    
    logger.info(f"Downloaded {count_base1 + count_base2} files")
    return count_base1 + count_base2


def upload_model_outputs():
    """Upload model outputs to Azure Blob"""
    logger.info("Uploading model outputs to Azure Blob...")
    
    blob_client = get_blob_client()
    
    # Upload reports/exports
    count = blob_client.upload_directory(
        container_name="model-outputs",
        local_directory="reports/exports",
        blob_prefix=""
    )
    
    logger.info(f"Uploaded {count} files")
    return count


def sync_all():
    """Full sync: download inputs, upload outputs"""
    logger.info("Starting full sync with Azure Blob...")
    
    # Download inputs
    downloaded = download_processed_data()
    logger.info(f"✓ Downloaded {downloaded} input files")
    
    # Upload outputs
    uploaded = upload_model_outputs()
    logger.info(f"✓ Uploaded {uploaded} output files")
    
    logger.info("Sync complete!")


def main():
    parser = argparse.ArgumentParser(description="Azure Blob Storage sync utilities")
    parser.add_argument(
        "command",
        choices=["download", "upload", "sync"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_processed_data()
    elif args.command == "upload":
        upload_model_outputs()
    elif args.command == "sync":
        sync_all()


if __name__ == "__main__":
    main()


