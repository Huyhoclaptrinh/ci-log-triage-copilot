import pandas as pd

def prepare_data(csv_path):
    """
    Loads and preprocesses the raw CI/CD log data from the input CSV file.
    """
    print("--- Preparing Data ---")
    df = pd.read_csv(csv_path)

    # Normalize status
    df["status_norm"] = (
        df["status"]
        .astype(str).str.normalize("NFKC")
        .str.strip().str.casefold()
    )

    STATUS_BUCKET = {
        "success": "success",
        "failed": "failure",
        "running": "running",
        "skipped": "skipped",
    }
    df["status_bucket"] = df["status_norm"].map(STATUS_BUCKET).fillna("other")
    df["is_failure"] = (df["status_bucket"] == "failure").astype(int)

    # Parse timestamps
    df["_ts"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S%z", errors="coerce")

    # Filter for failures
    fail_df = df[df["is_failure"] == 1].copy()
    print(f"Found {len(fail_df)} failure records.")
    return fail_df
