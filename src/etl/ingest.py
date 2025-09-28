from pathlib import Path
import pandas as pd
import yaml
import requests
import os
from urllib.parse import urlparse, parse_qs

def _to_month(dt_series):
    dt = pd.to_datetime(dt_series)
    return dt.dt.to_period('M').dt.to_timestamp()

def load_config(path="configs/settings.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_directories_exist(paths):
    """Ensure all required directories exist"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def _extract_google_drive_file_id(url):
    """Extract file ID from Google Drive sharing URL"""
    if '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    elif 'id=' in url:
        return url.split('id=')[1].split('&')[0]
    else:
        raise ValueError("Invalid Google Drive URL format")

def _download_google_drive_file(file_id, local_path):
    """Download file from Google Drive using file ID"""
    export_formats = [
        f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx",
        f"https://drive.google.com/uc?export=download&id={file_id}",
    ]
    print(f"Downloading file from Google Drive (ID: {file_id})...")
    for i, download_url in enumerate(export_formats):
        try:
            print(f"Trying download method {i+1}/{len(export_formats)}...")
            response = requests.get(download_url, allow_redirects=True, stream=True)
            content_type = response.headers.get('content-type', '').lower()
            if response.status_code == 200:
                if 'text/html' in content_type:
                    print(f"Method {i+1} returned HTML page, trying next method...")
                    continue
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                with open(local_path, 'rb') as f:
                    first_bytes = f.read(100)
                    if b'<!DOCTYPE html>' in first_bytes or b'<html>' in first_bytes:
                        print(f"Method {i+1} downloaded HTML instead of Excel file, trying next method...")
                        continue
                print(f"File downloaded successfully using method {i+1}: {local_path}")
                return str(local_path)
            else:
                print(f"Method {i+1} failed with status {response.status_code}, trying next method...")
                continue
        except requests.RequestException as e:
            print(f"Method {i+1} failed with error: {e}, trying next method...")
            continue
    raise Exception(f"Failed to download Excel file from Google Drive. The file might be a Google Sheets document that needs to be downloaded as Excel format. Please ensure the file is shared publicly or try downloading it manually and placing it in data/raw/ folder.")

def _download_file_if_needed(url_or_path, local_path):
    """Download file from URL if it doesn't exist locally"""
    local_file = Path(local_path)
    if local_file.exists():
        print(f"File already exists locally: {local_path}")
        return str(local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if 'drive.google.com' in url_or_path or 'docs.google.com' in url_or_path:
        file_id = _extract_google_drive_file_id(url_or_path)
        return _download_google_drive_file(file_id, local_path)
    elif url_or_path.startswith(('http://', 'https://')):
        print(f"Downloading file from: {url_or_path}")
        try:
            response = requests.get(url_or_path, stream=True)
            response.raise_for_status()
            with open(local_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"File downloaded successfully: {local_path}")
            return str(local_file)
        except requests.RequestException as e:
            raise Exception(f"Failed to download file from {url_or_path}: {e}")
    else:
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {url_or_path}")
        return str(local_file)

def ingest_bases():
    cfg = load_config()
    raw_path = cfg["data"]["raw_excel"]
    
    # Handle remote URLs (Google Drive, HTTP, etc.) or local paths
    if raw_path.startswith(('http://', 'https://')) or 'drive.google.com' in raw_path or 'docs.google.com' in raw_path:
        local_filename = "Challenge_FIAP_Bases.xlsx"
        if 'drive.google.com' not in raw_path and 'docs.google.com' not in raw_path:
            parsed_url = urlparse(raw_path)
            if parsed_url.path:
                local_filename = os.path.basename(parsed_url.path) or local_filename
        local_path = f"data/raw/{local_filename}"
        actual_file_path = _download_file_if_needed(raw_path, local_path)
    else:
        actual_file_path = raw_path
    
    # Ensure all required directories exist
    required_dirs = [
        "data/raw", "data/processed/base1", "data/processed/base2",
        "reports/exports", "reports/logs"
    ]
    _ensure_directories_exist(required_dirs)

    # ---- Base 1 (perfil/ID) ----
    b1 = pd.read_excel(actual_file_path, sheet_name="Base 1 - ID", dtype={"ID": str})
    b1["DT_REFE"] = _to_month(b1["DT_REFE"])
    for col in ["VL_FATU", "VL_SLDO"]:
        b1[col] = pd.to_numeric(b1[col], errors="coerce")
    out1 = Path(cfg["data"]["processed_base1"])
    out1.mkdir(parents=True, exist_ok=True)
    b1.to_parquet(out1 / "base1.parquet", index=False)

    # ---- Base 2 (transações) ----
    b2 = pd.read_excel(actual_file_path, sheet_name="Base 2 - Transações",
                       dtype={"ID_PGTO": str, "ID_RCBE": str})
    b2["DT_REFE"] = _to_month(b2["DT_REFE"])
    b2["VL"] = pd.to_numeric(b2["VL"], errors="coerce").fillna(0)
    # normalizar DS_TRAN para domínios
    dom = cfg["domains"]["ds_tran_map"]
    b2["DS_TRAN"] = b2["DS_TRAN"].astype(str).str.upper().str.strip()
    b2["DS_TRAN"] = b2["DS_TRAN"].map(dom).fillna(dom.get("default", "OUTROS"))

    out2 = Path(cfg["data"]["processed_base2"])
    out2.mkdir(parents=True, exist_ok=True)
    b2.to_parquet(out2 / "base2.parquet", index=False)

    return {"base1_rows": len(b1), "base2_rows": len(b2)}