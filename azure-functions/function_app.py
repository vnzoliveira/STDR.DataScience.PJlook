"""
Azure Function: xlsx to parquet converter
Triggered when .xlsx file is uploaded to blob storage
"""
import logging
import io
import os
from datetime import datetime
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import pandas as pd

app = func.FunctionApp()

# Configuration
STORAGE_CONNECTION_STRING = os.environ.get("STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

def _to_month(dt_series):
    """Convert datetime to month period"""
    dt = pd.to_datetime(dt_series)
    return dt.dt.to_period('M').dt.to_timestamp()

@app.blob_trigger(
    arg_name="inputblob",
    path="raw-data/{name}.xlsx",
    connection="STORAGE_CONNECTION_STRING"
)
def xlsx_to_parquet(inputblob: func.InputStream):
    """
    Process uploaded .xlsx files and convert to parquet
    """
    logging.info(f"Processing blob: {inputblob.name}")
    logging.info(f"Blob size: {inputblob.length} bytes")
    
    try:
        # Read Excel file from blob
        excel_data = inputblob.read()
        logging.info("Excel file read successfully")
        
        # Parse Base 1 (Profile/ID)
        logging.info("Parsing Base 1 - ID...")
        base1 = pd.read_excel(
            io.BytesIO(excel_data),
            sheet_name="Base 1 - ID",
            dtype={"ID": str}
        )
        
        # Transform Base 1
        base1["DT_REFE"] = _to_month(base1["DT_REFE"])
        for col in ["VL_FATU", "VL_SLDO"]:
            base1[col] = pd.to_numeric(base1[col], errors="coerce")
        
        logging.info(f"Base 1 processed: {len(base1)} rows")
        
        # Parse Base 2 (Transactions)
        logging.info("Parsing Base 2 - Transações...")
        base2 = pd.read_excel(
            io.BytesIO(excel_data),
            sheet_name="Base 2 - Transações",
            dtype={"ID_PGTO": str, "ID_RCBE": str}
        )
        
        # Transform Base 2
        base2["DT_REFE"] = _to_month(base2["DT_REFE"])
        base2["VL"] = pd.to_numeric(base2["VL"], errors="coerce").fillna(0)
        
        # Normalize DS_TRAN
        ds_tran_map = {
            "PIX": "PIX",
            "TED": "TED",
            "BOLETO": "BOLETO",
            "SISTEMICO": "OUTROS"
        }
        base2["DS_TRAN"] = base2["DS_TRAN"].astype(str).str.upper().str.strip()
        base2["DS_TRAN"] = base2["DS_TRAN"].map(ds_tran_map).fillna("OUTROS")
        
        logging.info(f"Base 2 processed: {len(base2)} rows")
        
        # Convert to parquet and upload
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Upload Base 1
        base1_parquet = base1.to_parquet(index=False)
        base1_blob_client = blob_service_client.get_blob_client(
            container="processed-data",
            blob=f"base1/base1_{timestamp}.parquet"
        )
        base1_blob_client.upload_blob(base1_parquet, overwrite=True)
        logging.info(f"Base 1 uploaded: base1_{timestamp}.parquet")
        
        # Upload Base 2
        base2_parquet = base2.to_parquet(index=False)
        base2_blob_client = blob_service_client.get_blob_client(
            container="processed-data",
            blob=f"base2/base2_{timestamp}.parquet"
        )
        base2_blob_client.upload_blob(base2_parquet, overwrite=True)
        logging.info(f"Base 2 uploaded: base2_{timestamp}.parquet")
        
        # Also save as "latest" for easy access
        base1_latest_client = blob_service_client.get_blob_client(
            container="processed-data",
            blob="base1/base1.parquet"
        )
        base1_latest_client.upload_blob(base1_parquet, overwrite=True)
        
        base2_latest_client = blob_service_client.get_blob_client(
            container="processed-data",
            blob="base2/base2.parquet"
        )
        base2_latest_client.upload_blob(base2_parquet, overwrite=True)
        
        logging.info("✅ Processing complete!")
        
        return {
            "status": "success",
            "base1_rows": len(base1),
            "base2_rows": len(base2),
            "timestamp": timestamp
        }
        
    except Exception as e:
        logging.error(f"❌ Error processing blob: {str(e)}")
        raise


@app.function_name(name="HealthCheck")
@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    return func.HttpResponse(
        '{"status": "healthy", "function": "xlsx-to-parquet"}',
        status_code=200,
        mimetype="application/json"
    )


