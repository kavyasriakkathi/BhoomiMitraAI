"""
BhoomiMitra AI — Local Mandi Price Ingestor (Runs from Indian Network)

Fetches daily mandi commodity prices from api.data.gov.in (Agmarknet)
and pushes them to BhoomiMitra on Render.
"""
import os
import sys
import httpx

def main():
    api_key = os.environ.get("DATA_GOV_API_KEY", "").strip()
    sync_secret = os.environ.get("MARKET_SYNC_SECRET", "").strip()
    app_url = os.environ.get("APP_URL", "https://bhoomimitra-ai-whatsapp.onrender.com").rstrip("/")

    if not api_key:
        api_key = input("Enter your DATA_GOV_API_KEY: ").strip()
    if not sync_secret:
        sync_secret = input("Enter your MARKET_SYNC_SECRET (or WHATSAPP_VERIFY_TOKEN): ").strip()

    if not api_key or not sync_secret:
        print("ERROR: Both DATA_GOV_API_KEY and sync secret are required.")
        sys.exit(1)

    base_api = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    states = ["Telangana", "Andhra Pradesh"]
    all_records = []

    print("\nConnecting to api.data.gov.in from your Indian connection...")
    with httpx.Client(timeout=30.0, headers=headers) as client:
        for state in states:
            print(f"Fetching records for state='{state}'...")
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": 500,
                "filters[state.keyword]": state,
            }
            try:
                resp = client.get(base_api, params=params)
                if resp.status_code == 200:
                    records = resp.json().get("records", [])
                    print(f"  -> Found {len(records)} records for {state}")
                    all_records.extend(records)
                else:
                    print(f"  -> Warning: Data.gov.in HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                print(f"  -> Error for {state}: {e}")

        if not all_records:
            print("No records retrieved. Exiting.")
            sys.exit(0)

        print(f"\nTotal records retrieved: {len(all_records)}")
        print(f"Pushing to Render ({app_url}/market/sync-internal)...")

        sync_endpoint = f"{app_url}/market/sync-internal"
        sync_headers = {
            "X-Sync-Token": sync_secret,
            "Content-Type": "application/json",
        }

        chunk_size = 100
        total_inserted = 0
        for i in range(0, len(all_records), chunk_size):
            chunk = all_records[i : i + chunk_size]
            try:
                resp = client.post(sync_endpoint, headers=sync_headers, json=chunk, timeout=60.0)
                if resp.status_code == 200:
                    data = resp.json()
                    inserted = data.get("records_inserted", 0)
                    total_inserted += inserted
                    print(f"  -> Batch {i // chunk_size + 1}: sent {len(chunk)} records, {inserted} inserted/updated")
                else:
                    print(f"  -> Batch {i // chunk_size + 1} failed (HTTP {resp.status_code}): {resp.text[:200]}")
            except Exception as e:
                print(f"  -> Batch {i // chunk_size + 1} error: {e}")

        print(f"\nSUCCESS! Synced {len(all_records)} total records ({total_inserted} newly inserted into PostgreSQL).")

if __name__ == "__main__":
    main()
