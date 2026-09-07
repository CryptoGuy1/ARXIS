#!/usr/bin/env python3
"""
Fetch the ARXIS dataset from your Google Drive.

This script connects to YOUR Google Drive using OAuth 2.0,
lists your files so you can find the dataset, and downloads it.

Usage:
    python retrain/fetch_from_drive.py

The first time you run it, a browser window will open asking you
to sign in to Google and grant access. After that, a saved token
keeps you authenticated.
"""

import os
import sys
import io
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "Gas_Sensors_Measurements.csv"
IMAGES_DIR = DATA_DIR / "Thermal Camera Images"

# OAuth client secret — keep this file private, never commit it
CLIENT_SECRET = Path.home() / "Downloads" / "aurora_oauth_client_secret.json"
TOKEN_PATH = PROJECT_ROOT / "retrain" / "token.json"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def authenticate():
    """Authenticate with Google Drive API."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    creds = None

    # Load saved token if available
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # Refresh or create new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[INFO] Refreshing access token...")
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET.exists():
                print(f"[ERROR] OAuth secret not found: {CLIENT_SECRET}")
                return None

            print("[INFO] Opening browser for Google sign-in...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for next run
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        print(f"[OK] Token saved to {TOKEN_PATH}")

    return creds


def list_files(service, query="", folder_id=None):
    """List files in Google Drive."""
    from googleapiclient.discovery import build

    q = query
    if folder_id:
        q = f"'{folder_id}' in parents" + (f" and {query}" if query else "")
    elif query:
        q = query

    results = (
        service.files()
        .list(
            q=q,
            pageSize=100,
            fields="nextPageToken, files(id, name, size, mimeType)",
        )
        .execute()
    )
    return results.get("files", [])


def download_file(service, file_id, output_path):
    """Download a file from Google Drive."""
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"\r[INFO] {output_path.name}: {int(status.progress() * 100)}%", end="")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Downloaded: {output_path} ({size_mb:.1f} MB)")


def download_folder_recursive(service, folder_id, folder_name, output_base):
    """Recursively download a folder and its contents from Google Drive."""
    output_base = Path(output_base)

    if folder_name:
        output_base = output_base / folder_name

    # List contents of the folder
    files = list_files(service, folder_id=folder_id)

    if not files:
        print(f"[INFO] Folder '{folder_name}' is empty or not found.")
        return

    print(f"[INFO] Downloading folder '{folder_name}' ({len(files)} items)...")

    for f in files:
        if f["mimeType"] == "application/vnd.google-apps.folder":
            # Recurse into subfolder
            download_folder_recursive(service, f["id"], f["name"], output_base.parent if folder_name else output_base)
        else:
            # Download file
            file_path = output_base / f["name"]
            if file_path.exists():
                print(f"[SKIP] Already exists: {file_path}")
                continue
            download_file(service, f["id"], file_path)


def main():
    from googleapiclient.discovery import build

    print("=" * 60)
    print("ARXIS — Google Drive Dataset Fetcher")
    print("=" * 60)

    # Authenticate
    creds = authenticate()
    if not creds:
        return 1

    service = build("drive", "v3", credentials=creds)

    # Check what's missing
    csv_exists = CSV_PATH.exists()
    images_exist = IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir())

    if csv_exists and images_exist:
        print("\n[OK] Both CSV and Thermal Camera Images already exist.")
        return 0

    # Show menu
    print("\nWhat would you like to fetch?")
    if not csv_exists:
        print("  1. Gas Sensors Measurements.csv (required)")
    else:
        print("  1. [ALREADY HAVE] Gas Sensors Measurements.csv")

    if not images_exist:
        print("  2. Thermal Camera Images folder (required for vision)")
    else:
        print("  2. [ALREADY HAVE] Thermal Camera Images")

    print("  3. Both CSV + Images")
    print("  q. Quit")

    choice = input("\nEnter choice: ").strip().lower()

    if choice == "q":
        return 0

    if choice == "1" and not csv_exists:
        fetch_csv(service)
    elif choice == "2" and not images_exist:
        fetch_images(service)
    elif choice == "3":
        if not csv_exists:
            fetch_csv(service)
        if not images_exist:
            fetch_images(service)
    else:
        print("Invalid choice or files already exist.")
        return 1

    # Verify
    print("\n" + "=" * 60)
    print("Verification:")
    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
        print(f"  [OK] CSV: {CSV_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"  [MISSING] CSV: {CSV_PATH}")

    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        print(f"  [OK] Images: {IMAGES_DIR}")
    else:
        print(f"  [MISSING] Images: {IMAGES_DIR}")

    return 0


def fetch_csv(service):
    """Search for and download the CSV file."""
    print("\n[INFO] Searching for 'Gas Sensors' CSV...")
    files = list_files(
        service,
        query="name contains 'Gas_Sensors' or name contains 'Gas Sensors' or name contains 'gas_sensors' or name contains 'Measurement' or name contains 'measurement'",
    )

    if not files:
        print("[INFO] No matching CSV found. Listing all your files...")
        files = list_files(service)
        if not files:
            print("[ERROR] No files found in your Google Drive.")
            return

    # Display results
    print(f"\n{'#':>3}  {'Name':<50} {'Size':>10} {'ID'}")
    print("-" * 90)
    for i, f in enumerate(files, 1):
        size = int(f.get("size", 0))
        size_str = f"{size / 1024 / 1024:.1f} MB" if size else "—"
        print(f"{i:>3}  {f['name']:<50} {size_str:>10} {f['id']}")

    # Let user choose
    print()
    while True:
        choice = input("Enter file number to download (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                break
        except ValueError:
            pass
        print("Invalid choice. Try again.")

    selected = files[idx]
    print(f"\n[INFO] Downloading: {selected['name']}")
    download_file(service, selected["id"], CSV_PATH)


def fetch_images(service):
    """Search for and download the Thermal Camera Images folder."""
    print("\n[INFO] Searching for 'Thermal Camera Images' folder...")

    # Search for folders containing "Thermal" or "Camera" or "Images"
    folders = list_files(
        service,
        query="(name contains 'Thermal' or name contains 'Camera' or name contains 'Images') and mimeType = 'application/vnd.google-apps.folder'",
    )

    if not folders:
        print("[INFO] No matching folder found. Searching all folders...")
        folders = list_files(service, query="mimeType = 'application/vnd.google-apps.folder'")
        if not folders:
            print("[ERROR] No folders found in your Google Drive.")
            return

    # Display results
    print(f"\n{'#':>3}  {'Folder Name':<50} {'ID'}")
    print("-" * 80)
    for i, f in enumerate(folders, 1):
        print(f"{i:>3}  {f['name']:<50} {f['id']}")

    # Let user choose
    print()
    while True:
        choice = input("Enter folder number to download (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                break
        except ValueError:
            pass
        print("Invalid choice. Try again.")

    selected = folders[idx]
    print(f"\n[INFO] Downloading folder: {selected['name']}")
    download_folder_recursive(service, selected["id"], selected["name"], DATA_DIR)


if __name__ == "__main__":
    sys.exit(main())
