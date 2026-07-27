"""ClinVar data loading and processing.

Downloads ClinVar VCF, parses variants, and prepares training data
with Pathogenic/Benign labels for the meta-learner.

ClinVar classification mapping:
    Label 1 (Pathogenic): "Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"
    Label 0 (Benign): "Benign", "Likely benign", "Benign/Likely benign"
    Excluded: "Uncertain significance", "Conflicting", "not provided"
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from vus_meta.schemas import Variant, VariantType

CLINVAR_VCF_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
)

PATHOGENIC_TERMS = {
    "Pathogenic",
    "Likely_pathogenic",
    "Pathogenic/Likely_pathogenic",
}

BENIGN_TERMS = {
    "Benign",
    "Likely_benign",
    "Benign/Likely_benign",
}


def download_clinvar(output_dir: str | Path, force: bool = False) -> Path:
    """Download the latest ClinVar VCF file.

    Args:
        output_dir: Directory to save the file
        force: If True, re-download even if file exists

    Returns:
        Path to the downloaded VCF file
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clinvar.vcf.gz"

    if output_path.exists() and not force:
        logger.info(f"ClinVar VCF already exists: {output_path}")
        return output_path

    logger.info(f"Downloading ClinVar VCF from {CLINVAR_VCF_URL}")
    response = requests.get(CLINVAR_VCF_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                logger.info(f"  Downloaded {downloaded / 1e6:.0f} MB / {total_size / 1e6:.0f} MB")

    logger.info(f"ClinVar VCF saved to {output_path}")
    return output_path


def _parse_info_field(info_str: str) -> dict[str, str]:
    """Parse VCF INFO field into a dictionary."""
    info = {}
    for item in info_str.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = "True"
    return info


def _determine_variant_type(info: dict) -> VariantType:
    """Determine variant type from ClinVar molecular consequence."""
    mc = info.get("MC", "")
    clnvc = info.get("CLNVC", "")

    if "missense_variant" in mc:
        return VariantType.MISSENSE
    elif "nonsense" in mc or "stop_gained" in mc:
        return VariantType.NONSENSE
    elif "synonymous_variant" in mc:
        return VariantType.SYNONYMOUS
    elif "splice" in mc:
        return VariantType.SPLICE
    elif "frameshift" in mc:
        return VariantType.FRAMESHIFT
    elif "inframe" in mc:
        return VariantType.INFRAME_INDEL
    elif "intron_variant" in mc:
        return VariantType.INTRONIC
    elif "UTR" in mc:
        return VariantType.UTR
    else:
        return VariantType.UNKNOWN


def parse_clinvar_vcf(
    vcf_path: str | Path,
    variant_types: Optional[list[VariantType]] = None,
    max_variants: Optional[int] = None,
) -> pd.DataFrame:
    """Parse ClinVar VCF into a DataFrame of labeled variants.

    Args:
        vcf_path: Path to clinvar.vcf.gz
        variant_types: If specified, only include these variant types
        max_variants: Maximum number of variants to parse

    Returns:
        DataFrame with columns: chrom, pos, ref, alt, gene, clnsig,
        label (1=path, 0=benign, -1=vus), variant_type, hgvs_c, hgvs_p, etc.
    """
    vcf_path = Path(vcf_path)
    records = []
    count = 0

    opener = gzip.open if str(vcf_path).endswith(".gz") else open

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos, _, ref, alt, _, _, info_str = fields[:8]

            # Skip multi-allelic (simplified; in production we'd split)
            if "," in alt:
                alt = alt.split(",")[0]

            info = _parse_info_field(info_str)
            clnsig = info.get("CLNSIG", "")

            # Assign label
            if any(term in clnsig for term in PATHOGENIC_TERMS):
                label = 1
            elif any(term in clnsig for term in BENIGN_TERMS):
                label = 0
            else:
                label = -1  # VUS or other

            # Determine variant type
            vtype = _determine_variant_type(info)
            if variant_types and vtype not in variant_types:
                continue

            # Extract gene name
            gene = info.get("GENEINFO", "").split(":")[0] if "GENEINFO" in info else ""

            # Extract HGVS
            clnhgvs = info.get("CLNHGVS", "")

            records.append({
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "gene": gene,
                "clnsig": clnsig,
                "label": label,
                "variant_type": vtype.value,
                "review_status": info.get("CLNREVSTAT", ""),
                "clnhgvs": clnhgvs,
                "clinvar_id": info.get("CLNVARID", ""),
                "origin": info.get("ORIGIN", ""),
                "mc": info.get("MC", ""),
            })

            count += 1
            if max_variants and count >= max_variants:
                break

    df = pd.DataFrame(records)
    logger.info(
        f"Parsed {len(df)} ClinVar variants: "
        f"{(df['label'] == 1).sum()} pathogenic, "
        f"{(df['label'] == 0).sum()} benign, "
        f"{(df['label'] == -1).sum()} VUS/other"
    )
    return df


def prepare_training_data(
    clinvar_df: pd.DataFrame,
    min_review_stars: int = 1,
    exclude_conflicting: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training and VUS evaluation sets from ClinVar data.

    Args:
        clinvar_df: DataFrame from parse_clinvar_vcf()
        min_review_stars: Minimum ClinVar review status stars
        exclude_conflicting: Whether to exclude conflicting interpretations

    Returns:
        (training_df, vus_df) — training has label in {0, 1}, vus has label=-1
    """
    # Map review status to star count
    star_map = {
        "practice_guideline": 4,
        "reviewed_by_expert_panel": 3,
        "criteria_provided,_multiple_submitters,_no_conflicts": 2,
        "criteria_provided,_conflicting_interpretations": 1,
        "criteria_provided,_single_submitter": 1,
        "no_assertion_criteria_provided": 0,
        "no_assertion_provided": 0,
    }

    df = clinvar_df.copy()
    df["stars"] = df["review_status"].map(
        lambda x: max(
            (star_map.get(status.strip(), 0) for status in x.split(",")),
            default=0,
        )
    )

    # Filter training data
    training_mask = (
        (df["label"].isin([0, 1]))
        & (df["stars"] >= min_review_stars)
    )
    if exclude_conflicting:
        training_mask &= ~df["review_status"].str.contains("conflicting", case=False, na=False)

    training_df = df[training_mask].copy()
    vus_df = df[df["label"] == -1].copy()

    logger.info(
        f"Training set: {len(training_df)} variants "
        f"({(training_df['label'] == 1).sum()} P, {(training_df['label'] == 0).sum()} B)"
    )
    logger.info(f"VUS set: {len(vus_df)} variants")

    return training_df, vus_df


def clinvar_df_to_variants(df: pd.DataFrame) -> list[Variant]:
    """Convert a ClinVar DataFrame to a list of Variant objects."""
    variants = []
    for _, row in df.iterrows():
        v = Variant(
            chrom=str(row["chrom"]),
            pos=int(row["pos"]),
            ref=str(row["ref"]),
            alt=str(row["alt"]),
            gene=str(row.get("gene", "")),
            variant_type=VariantType(row.get("variant_type", "unknown")),
        )
        variants.append(v)
    return variants
