import os, json, pandas as pd

def ensure_full_stock_alias():
    alias_file = os.getenv("STOCK_ALIAS_FILE", "stock_alias.json")
    if os.path.exists(alias_file):
        with open(alias_file, "r", encoding="utf-8") as f:
            return json.load(f)

    urls = {
        "nasdaq": "https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_json/data/a09b7bbf34d2b57e7c877785f6e9f67a/nasdaq-listed_json.json",
        "nyse": "https://pkgstore.datahub.io/core/nyse-other-listings/nyse-listed_json/data/9177e49df77bb768a2602944d9bb0dbf/nyse-listed_json.json"
    }

    alias = {}
    for url in urls.values():
        df = pd.read_json(url)
        for _, row in df.iterrows():
            alias[row["Company Name"].strip().lower()] = row["Symbol"].strip().upper()

    with open(alias_file, "w", encoding="utf-8") as f:
        json.dump(alias, f, indent=2)
    return alias