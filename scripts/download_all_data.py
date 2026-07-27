#!/usr/bin/env python3
"""Download all required data for VUS Meta-Predictor.

Run once before using the tool:
    python scripts/download_all_data.py

Or via Docker:
    docker compose run --rm download
"""

import os
import sys
import requests
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def download(url, path, verify=True):
    """Download a file with progress."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1000:
        logger.info(f"  Already exists: {path.name} ({path.stat().st_size/(1024**2):.0f} MB)")
        return True
    
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading {path.name}...")
    
    try:
        r = requests.get(url, stream=True, timeout=60, verify=verify)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        dl = 0
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                dl += len(chunk)
                if total and dl % (50*1024*1024) < 1024*1024:
                    logger.info(f"    {dl/(1024**2):.0f} / {total/(1024**2):.0f} MB")
        logger.info(f"  ✅ {path.name}: {path.stat().st_size/(1024**2):.1f} MB")
        return True
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  VUS Meta-Predictor — Data Download")
    print("=" * 60)
    
    results = {}
    
    # 1. ClinVar
    print("\n📦 ClinVar (variant database)...")
    results['clinvar_vcf'] = download(
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
        "data/raw/clinvar.vcf.gz")
    results['clinvar_tbi'] = download(
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi",
        "data/raw/clinvar.vcf.gz.tbi")
    results['variant_summary'] = download(
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        "data/raw/variant_summary.txt.gz")
    
    # 2. AlphaMissense
    print("\n📦 AlphaMissense (DeepMind pathogenicity scores)...")
    results['alphamissense'] = download(
        "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz",
        "data/raw/AlphaMissense_hg38.tsv.gz")
    
    # 3. Conservation scores
    print("\n📦 Conservation scores (UCSC)...")
    results['phylop'] = download(
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw",
        "data/raw/hg38.phyloP100way.bw", verify=False)
    results['phastcons'] = download(
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw",
        "data/raw/hg38.phastCons100way.bw", verify=False)
    
    # 4. HPO (phenotype ontology)
    print("\n📦 HPO (Human Phenotype Ontology)...")
    results['hpo_obo'] = download(
        "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo",
        "data/raw/hp.obo")
    results['genes_to_pheno'] = download(
        "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
        "data/raw/genes_to_phenotype.txt")
    
    # Summary
    print("\n" + "=" * 60)
    ok = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  Downloaded {ok}/{total} files")
    
    total_size = sum(
        Path(f"data/raw/{f}").stat().st_size 
        for f in os.listdir("data/raw") 
        if Path(f"data/raw/{f}").is_file()
    ) / (1024**3)
    print(f"  Total data size: {total_size:.1f} GB")
    
    if ok < total:
        print(f"\n  ⚠️  Some downloads failed. Re-run to retry.")
    else:
        print(f"\n  ✅ All data ready! Run the pipeline:")
        print(f"     python scripts/train_real_clinvar.py")
        print(f"     python scripts/esm2_scoring.py")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
