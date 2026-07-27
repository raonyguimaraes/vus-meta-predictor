"""Command-line interface for VUS Meta-Predictor."""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger


@click.group()
@click.version_option(version="0.1.0")
def main():
    """VUS Meta-Predictor: Variant interpretation with genomic language models."""
    pass


@main.command()
@click.argument("vcf_path", type=click.Path(exists=True))
@click.option("--model", "-m", type=click.Path(exists=True), required=True,
              help="Path to trained model directory")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output TSV path")
@click.option("--genome", "-g", type=click.Path(exists=True), default=None,
              help="Reference genome FASTA")
def predict(vcf_path: str, model: str, output: str | None, genome: str | None):
    """Score variants in a VCF file."""
    from vus_meta.core import VUSPredictor

    predictor = VUSPredictor.from_pretrained(model)
    predictor.setup()

    if output is None:
        output = str(Path(vcf_path).stem) + ".scored.tsv"

    df = predictor.predict_vcf(vcf_path, output_path=output, genome_fasta=genome)

    click.echo(f"\nScored {len(df)} variants → {output}")
    click.echo(f"\nClassification summary:")
    click.echo(df["classification"].value_counts().to_string())


@main.command()
@click.option("--clinvar", type=click.Path(exists=True), required=True,
              help="Path to ClinVar VCF")
@click.option("--dbnsfp", type=click.Path(exists=True), default=None,
              help="Path to dbNSFP file")
@click.option("--output", "-o", type=click.Path(), default="models/meta_v1",
              help="Output model directory")
@click.option("--folds", type=int, default=5, help="Number of CV folds")
def train(clinvar: str, dbnsfp: str | None, output: str, folds: int):
    """Train the meta-learner on ClinVar data."""
    from vus_meta.data.clinvar import parse_clinvar_vcf, prepare_training_data
    from vus_meta.core import VUSPredictor
    from vus_meta.data.features import FeatureExtractor
    from vus_meta.predictors.dbnsfp import DbNSFPPredictor

    click.echo("Parsing ClinVar...")
    clinvar_df = parse_clinvar_vcf(clinvar)
    training_df, vus_df = prepare_training_data(clinvar_df)

    click.echo("Setting up feature extraction...")
    predictors = []
    if dbnsfp:
        predictors.append(DbNSFPPredictor(dbnsfp))

    extractor = FeatureExtractor(predictors=predictors)
    extractor.setup()

    predictor = VUSPredictor(feature_extractor=extractor)

    # Extract features and train
    from vus_meta.data.clinvar import clinvar_df_to_variants
    variants = clinvar_df_to_variants(training_df)
    X, feature_names = extractor.extract_batch(variants)
    y = training_df["label"].values

    click.echo(f"Training on {X.shape[0]} variants with {X.shape[1]} features...")
    metrics = predictor.train(X, y, feature_names, cross_validate=True, n_folds=folds)

    predictor.save(output)
    click.echo(f"\nModel saved to {output}")
    click.echo(f"CV AUROC: {metrics['cv_auroc']:.4f}")
    click.echo(f"CV AUPRC: {metrics['cv_auprc']:.4f}")


@main.command()
@click.option("--output", "-o", type=click.Path(), default="data/raw",
              help="Download directory")
def download_data(output: str):
    """Download ClinVar and other reference data."""
    from vus_meta.data.clinvar import download_clinvar

    click.echo("Downloading ClinVar VCF...")
    path = download_clinvar(output)
    click.echo(f"Downloaded: {path}")


@main.command()
@click.argument("hgvs", type=str)
@click.option("--model", "-m", type=click.Path(exists=True), required=True,
              help="Path to trained model directory")
def query(hgvs: str, model: str):
    """Score a single variant by HGVS notation.

    Example: vus-predict query "NM_000546.6:c.215C>G" -m models/meta_v1
    """
    from vus_meta.core import VUSPredictor

    predictor = VUSPredictor.from_pretrained(model)
    predictor.setup()

    # Parse HGVS to variant (simplified)
    from vus_meta.schemas import Variant
    variant = Variant(chrom="?", pos=0, ref="?", alt="?", hgvs_c=hgvs)

    result = predictor.predict_variant(variant)
    click.echo(f"\n{result}")
    click.echo(f"\nTop contributing features:")
    for name, value in sorted(
        result.feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
    )[:10]:
        direction = "↑ pathogenic" if value > 0 else "↓ benign"
        click.echo(f"  {name}: {value:+.4f} ({direction})")


if __name__ == "__main__":
    main()
