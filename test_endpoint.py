import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
# Configuration
data_path = "data/future_unseen_examples.csv"
API_KEY = os.getenv("API_KEY", default="NO_KEY_AVAILABLE")
url = "http://localhost:8000/predict"


def submit_examples(csv_path, url, API_KEY, max_examples=5):
    """
    Read examples from CSV and submit to the API endpoint. Prints status and response JSON.
    """
    # Load unseen examples
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    # Submit up to max_examples
    for i, record in enumerate(records[:max_examples]):
        try:
            response = requests.post(
                f"{url}?api_key={API_KEY}",
                json=record,
                timeout=5
            )
            print(f"Example {i + 1} | Status: {response.status_code}")
            print(response.json())
        except Exception as e:
            print(f"Example {i + 1} | Error: {e}")


def main():
    submit_examples(data_path, url, API_KEY)


if __name__ == "__main__":
    main()
