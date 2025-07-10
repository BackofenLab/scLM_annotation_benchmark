#!/usr/bin/env python3

import argparse
import os
import requests
from tqdm import tqdm

def get_dataset_url(subtype):
    article_id = 27921984
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"

    r = requests.get(api_url)
    r.raise_for_status()

    files = r.json()['files']
    for f in files:
        name = f['name'].split("_TSP1")[0]
        if name.lower() == subtype.lower():
            return f['download_url']
    raise ValueError(f"Could not find dataset: {subtype}")

def download_with_tqdm(url, output_path, chunk_size=1024):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    progress = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path))

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))

    progress.close()

def download_Tabula_sapiens(output_path, subtype):
    url = get_dataset_url(subtype)
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    download_with_tqdm(url, output_path)

def main():
    parser = argparse.ArgumentParser(description="Download single-cell dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name (currently: Tabula_sapiens or HIHA)")
    parser.add_argument("--subtype", required=True, help="Subtype, either cell type for HIHA or body site for Tabula sapiens")
    parser.add_argument("--out", required=True, help="Output path")

    args = parser.parse_args()
    if args.dataset == "Tabula_sapiens":
        download_Tabula_sapiens(args.out, args.subtype)
    elif args.dataset == "HIHA":
        raise NotImplementedError("HIHA download is not supported yet")
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

if __name__ == "__main__":
    main()
