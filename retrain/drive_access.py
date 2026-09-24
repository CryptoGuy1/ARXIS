"""
ARXIS — Google Drive direct access module.

Lets the system read the CSV and images directly from your Google Drive
without downloading the whole dataset. Uses OAuth 2.0 for authentication.

Usage in main.py / app.py:
    from retrain.drive_access import DriveAccessor
    drive = DriveAccessor()
    df = drive.read_csv('Gas_Sensors_Measurements.csv')
"""

import io
import sys
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIENT_SECRET = Path.home() / "Downloads" / "aurora_oauth_client_secret.json"
TOKEN_PATH = PROJECT_ROOT / "retrain" / "token.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class DriveAccessor:
    """Read files and images directly from Google Drive."""

    def __init__(self):
        self._service = None
        self._creds = None

    @property
    def service(self):
        if self._service is None:
            self._service = self._connect()
        return self._service

    def _connect(self):
        """Authenticate and build Drive service."""
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        creds = None

        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not CLIENT_SECRET.exists():
                    raise FileNotFoundError(
                        f"OAuth secret not found: {CLIENT_SECRET}"
                    )
                print("[INFO] Opening browser for Google sign-in...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CLIENT_SECRET), SCOPES
                )
                creds = flow.run_local_server(port=0)

            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())

        self._creds = creds
        return build("drive", "v3", credentials=creds)

    def _find_file(self, name_contains: str, mime_type: str = None):
        """Find a file by name in Google Drive."""
        query = " or ".join([f"name contains '{n}'" for n in name_contains.split("|")])
        if mime_type:
            query += f" and mimeType = '{mime_type}'"

        results = self.service.files().list(
            q=query, pageSize=10,
            fields="files(id, name, size, mimeType)"
        ).execute()

        return results.get("files", [])

    def read_csv(self, name: str = "Gas_Sensors_Measurements.csv") -> "pd.DataFrame":
        """Read a CSV file directly from Google Drive into pandas."""
        import pandas as pd
        from googleapiclient.http import MediaIoBaseDownload

        # Find the file
        candidates = self._find_file("Gas_Sensors|Gas Sensors|gas_sensors|Measurement")
        if not candidates:
            raise FileNotFoundError(f"File '{name}' not found in Google Drive")

        file_id = candidates[0]["id"]
        print(f"[DRIVE] Streaming from Google Drive: {candidates[0]['name']}")

        # Stream into memory
        request = self.service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"\r[DRIVE] Downloading: {int(status.progress() * 100)}%", end="")
        print()

        buffer.seek(0)
        df = pd.read_csv(buffer)
        print(f"[DRIVE] Loaded {len(df)} rows")
        return df

    def list_image_folders(self):
        """Find image folders in Google Drive."""
        folders = self._find_file(
            "Thermal|Camera|Images",
            mime_type="application/vnd.google-apps.folder"
        )
        return folders

    def list_images_in_folder(self, folder_id: str):
        """List all image files in a Drive folder."""
        query = f"'{folder_id}' in parents and (mimeType contains 'image/' or name contains '.png' or name contains '.jpg')"
        results = self.service.files().list(
            q=query, pageSize=1000,
            fields="files(id, name, size, mimeType)"
        ).execute()
        return results.get("files", [])

    def read_image(self, file_id: str) -> bytes:
        """Read an image file from Drive into memory."""
        from googleapiclient.http import MediaIoBaseDownload

        request = self.service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        return buffer.getvalue()

    def download_file(self, file_id: str, output_path: str):
        """Download a single file to local disk."""
        from googleapiclient.http import MediaIoBaseDownload

        request = self.service.files().get_media(fileId=file_id)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"\r{output_path.name}: {int(status.progress() * 100)}%", end="")
        print()


def get_drive_accessor() -> DriveAccessor:
    """Get the shared DriveAccessor instance."""
    return DriveAccessor()


if __name__ == "__main__":
    # Quick test
    drive = DriveAccessor()
    print("[DRIVE] Authenticated successfully")
    print("[DRIVE] Searching for CSV...")
    files = drive._find_file("Gas_Sensors|Gas Sensors|Measurement")
    for f in files:
        print(f"  - {f['name']} ({int(f.get('size', 0)) / 1024 / 1024:.1f} MB)")
