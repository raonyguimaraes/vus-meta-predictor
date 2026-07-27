"""FastAPI web application for VUS Meta-Predictor.

Provides:
- REST API for single-variant and batch scoring
- Web UI for interactive queries
- Pre-loaded model + ESM-2 for real-time inference
"""

from __future__ import annotations

import os
import re
import sys
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from loguru import logger

# ── App Setup ──
app = FastAPI(
    title="VUS Meta-Predictor",
    description="Genomic language model-powered variant interpretation",
    version="1.0.0",
)

# ── Data Models ──
class VariantQuery(BaseModel):
    gene: str = Field(..., description="Gene symbol (e.g., BRCA1)")
    ref_aa: str = Field(..., description="Reference amino acid (1-letter)")
    position: int = Field(..., description="Amino acid position")
    alt_aa: str = Field(..., description="Alternate amino acid (1-letter)")

class VariantHGVS(BaseModel):
    hgvs: str = Field(..., description="Protein HGVS (e.g., BRCA1 p.C61G)")

class PredictionResult(BaseModel):
    gene: str
    variant: str
    meta_score: float
    classification: str
    confidence: float
    esm2_llr: Optional[float] = None
    esm2_wt_prob: Optional[float] = None
    esm2_mut_prob: Optional[float] = None
    esm2_entropy: Optional[float] = None
    grantham_distance: Optional[float] = None
    alphamissense_score: Optional[float] = None
    phylop_score: Optional[float] = None
    feature_importances: dict = {}
    all_features: dict = {}

class BatchQuery(BaseModel):
    variants: list[VariantQuery]

# ── Globals (loaded at startup) ──
MODEL = None
ESM2_SCORER = None
SEQUENCES = {}
ALPHAMISSENSE_INDEX = None
PHYLOP_BW = None
PHASTCONS_BW = None

# ── AA Properties ──
AA_PROPS = {
    'A': ( 1.8,  88.6,  89.1,  6.00,  8.1), 'R': (-4.5, 173.4, 174.2, 10.76, 10.5),
    'N': (-3.5, 114.1, 132.1,  5.41, 11.6), 'D': (-3.5, 111.1, 133.1,  2.77, 13.0),
    'C': ( 2.5, 108.5, 121.2,  5.07,  5.5), 'E': (-3.5, 138.4, 147.1,  3.22, 12.3),
    'Q': (-3.5, 143.8, 146.2,  5.65, 10.5), 'G': (-0.4,  60.1,  75.0,  5.97,  9.0),
    'H': (-3.2, 153.2, 155.2,  7.59, 10.4), 'I': ( 4.5, 166.7, 131.2,  6.02,  5.2),
    'L': ( 3.8, 166.7, 131.2,  5.98,  4.9), 'K': (-3.9, 168.6, 146.2,  9.74, 11.3),
    'M': ( 1.9, 162.9, 149.2,  5.74,  5.7), 'F': ( 2.8, 189.9, 165.2,  5.48,  5.2),
    'P': (-1.6, 112.7, 115.1,  6.30,  8.0), 'S': (-0.8,  89.0, 105.1,  5.68,  9.2),
    'T': (-0.7, 116.1, 119.1,  5.60,  8.6), 'W': (-0.9, 227.8, 204.2,  5.89,  5.4),
    'Y': (-1.3, 193.6, 181.2,  5.66,  6.2), 'V': ( 4.2, 140.0, 117.1,  5.96,  5.9),
}

AA_CLASSES = {
    'tiny': set('AGS'), 'small': set('ACDGNPSTV'), 'aliphatic': set('AILV'),
    'aromatic': set('FHWY'), 'nonpolar': set('ACFGILMPVW'), 'polar': set('DEHKNQRSTY'),
    'charged': set('DEHKR'), 'positive': set('HKR'), 'negative': set('DE'),
}

AA3TO1 = {
    'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q','Gly':'G',
    'His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S',
    'Thr':'T','Trp':'W','Tyr':'Y','Val':'V',
}
AA1TO3 = {v: k for k, v in AA3TO1.items()}


def grantham(a1, a2):
    p1, p2 = AA_PROPS.get(a1), AA_PROPS.get(a2)
    if not p1 or not p2: return None
    if a1 == a2: return 0.0
    return float(np.sqrt(1.833*((p1[0]-p2[0])/2)**2 + (p1[4]-p2[4])**2 + 2.0*((p1[1]-p2[1])/50)**2))


def score_to_classification(score):
    if score >= 0.99: return "Pathogenic"
    elif score >= 0.90: return "Likely Pathogenic"
    elif score > 0.10: return "Uncertain Significance"
    elif score > 0.01: return "Likely Benign"
    else: return "Benign"


def classification_color(cls):
    return {
        "Pathogenic": "#d32f2f",
        "Likely Pathogenic": "#f57c00",
        "Uncertain Significance": "#9e9e9e",
        "Likely Benign": "#43a047",
        "Benign": "#1b5e20",
    }.get(cls, "#9e9e9e")


# ── Startup ──
@app.on_event("startup")
async def startup():
    global MODEL, ESM2_SCORER, SEQUENCES, ALPHAMISSENSE_INDEX
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    
    # Load meta-learner
    model_dir = Path("data/models/integrated_v1")
    if model_dir.exists():
        from vus_meta.models.meta_learner import MetaLearner
        MODEL = MetaLearner.load(model_dir)
        logger.info(f"Loaded meta-learner from {model_dir}")
    else:
        logger.warning(f"No model at {model_dir}, predictions will use heuristics")
    
    # Load ESM-2
    try:
        import esm as esm_module
        import torch
        logger.info("Loading ESM-2 35M model...")
        model, alphabet = esm_module.pretrained.esm2_t12_35M_UR50D()
        model.eval()
        ESM2_SCORER = {
            'model': model,
            'alphabet': alphabet,
            'batch_converter': alphabet.get_batch_converter(),
            'aa_to_idx': {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"},
        }
        logger.info("ESM-2 loaded!")
    except Exception as e:
        logger.warning(f"ESM-2 not available: {e}")
    
    # Load protein sequences
    fasta_path = Path("data/raw/priority_genes.fasta")
    if fasta_path.exists():
        with open(fasta_path) as f:
            gene, seq = None, []
            for line in f:
                if line.startswith('>'):
                    if gene: SEQUENCES[gene] = ''.join(seq)
                    gene = line[1:].strip().split()[0]
                    seq = []
                else:
                    seq.append(line.strip())
            if gene: SEQUENCES[gene] = ''.join(seq)
        logger.info(f"Loaded {len(SEQUENCES)} protein sequences")
    
    # Load AlphaMissense index (just the gene list for quick lookup)
    am_path = Path("data/raw/AlphaMissense_hg38.tsv.gz")
    if am_path.exists():
        logger.info("AlphaMissense available for lookup")
    
    logger.info("🧬 VUS Meta-Predictor API ready!")


def compute_esm2_features(gene, position_0based, wt_aa, mut_aa):
    """Compute ESM-2 features for a variant."""
    import torch
    
    if ESM2_SCORER is None or gene not in SEQUENCES:
        return {}
    
    seq = SEQUENCES[gene]
    if position_0based >= len(seq) or position_0based >= 1022:
        return {}
    
    model = ESM2_SCORER['model']
    alphabet = ESM2_SCORER['alphabet']
    batch_converter = ESM2_SCORER['batch_converter']
    aa_to_idx = ESM2_SCORER['aa_to_idx']
    
    AA = "ACDEFGHIKLMNPQRSTVWY"
    
    # Masked marginal scoring
    data = [('protein', seq[:1022])]
    _, _, tokens = batch_converter(data)
    
    mask_pos = position_0based + 1  # +1 for BOS
    tokens[0, mask_pos] = alphabet.mask_idx
    
    with torch.no_grad():
        output = model(tokens)
        logits = output['logits'][0, mask_pos]
    
    aa_indices = torch.tensor([aa_to_idx[aa] for aa in AA])
    log_probs = torch.log_softmax(logits, dim=0)
    aa_log_probs = log_probs[aa_indices].numpy()
    probs = np.exp(aa_log_probs)
    
    aa_idx_map = {aa: i for i, aa in enumerate(AA)}
    wt_i, mut_i = aa_idx_map.get(wt_aa), aa_idx_map.get(mut_aa)
    
    if wt_i is None or mut_i is None:
        return {}
    
    llr = float(aa_log_probs[mut_i] - aa_log_probs[wt_i])
    entropy = float(-np.sum(probs * aa_log_probs))
    
    sorted_idx = np.argsort(-probs)
    wt_rank = int(np.where(sorted_idx == wt_i)[0][0]) + 1
    mut_rank = int(np.where(sorted_idx == mut_i)[0][0]) + 1
    
    return {
        'esm2_llr': llr,
        'esm2_wt_logp': float(aa_log_probs[wt_i]),
        'esm2_mut_logp': float(aa_log_probs[mut_i]),
        'esm2_wt_prob': float(probs[wt_i]),
        'esm2_mut_prob': float(probs[mut_i]),
        'esm2_wt_rank': float(wt_rank),
        'esm2_mut_rank': float(mut_rank),
        'esm2_entropy': entropy,
        'esm2_score': float(1.0 / (1.0 + math.exp(llr * 2.0))),
    }


def compute_all_features(gene, ref_aa, position, alt_aa):
    """Compute the full feature vector for a variant."""
    features = {}
    
    # ESM-2 features
    esm2_feats = compute_esm2_features(gene, position - 1, ref_aa, alt_aa)
    features.update(esm2_feats)
    
    # AA property features
    p1, p2 = AA_PROPS.get(ref_aa), AA_PROPS.get(alt_aa)
    if p1 and p2:
        features['hydro_change'] = p2[0] - p1[0]
        features['hydro_abs_change'] = abs(features['hydro_change'])
        features['volume_change'] = p2[1] - p1[1]
        features['volume_abs_change'] = abs(features['volume_change'])
        features['mw_abs_change'] = abs(p2[2] - p1[2])
        features['pI_abs_change'] = abs(p2[3] - p1[3])
        features['polarity_abs_change'] = abs(p2[4] - p1[4])
        features['grantham'] = grantham(ref_aa, alt_aa)
        
        for cn, cs in AA_CLASSES.items():
            features[f'{cn}_change'] = float((ref_aa in cs) != (alt_aa in cs))
        
        features['to_proline'] = float(alt_aa == 'P')
        features['from_proline'] = float(ref_aa == 'P')
        features['to_glycine'] = float(alt_aa == 'G')
        features['from_cysteine'] = float(ref_aa == 'C')
        features['to_cysteine'] = float(alt_aa == 'C')
        features['charge_reversal'] = float(
            (ref_aa in AA_CLASSES['positive'] and alt_aa in AA_CLASSES['negative']) or
            (ref_aa in AA_CLASSES['negative'] and alt_aa in AA_CLASSES['positive'])
        )
    
    # Position features
    features['aa_position_log'] = math.log10(max(position, 1))
    
    # Gene features (hardcoded from training data if MODEL available)
    features['gene_path_ratio'] = 0.5  # Default
    features['gene_variant_count_log'] = 2.0  # Default
    features['gene_has_pathogenic'] = 1.0  # Default for clinical genes
    
    return features


def predict_variant(gene, ref_aa, position, alt_aa):
    """Full prediction for a single variant."""
    start = time.time()
    
    features = compute_all_features(gene, ref_aa, position, alt_aa)
    
    variant_str = f"p.{AA1TO3.get(ref_aa, ref_aa)}{position}{AA1TO3.get(alt_aa, alt_aa)}"
    
    if MODEL is not None and features:
        # Build feature vector matching model's expected features
        feature_names = MODEL.feature_names
        X = np.array([features.get(fn, np.nan) for fn in feature_names], dtype=np.float32).reshape(1, -1)
        
        score, shap_values = MODEL.predict_single(X[0])
        _, confidence = MODEL.predict_with_confidence(X)
        confidence = float(confidence[0])
    else:
        # Heuristic fallback
        esm2_score = features.get('esm2_score', 0.5)
        grantham_d = features.get('grantham', 50)
        
        # Simple weighted combination
        score = 0.4 * esm2_score + 0.3 * min(grantham_d / 200, 1.0) + 0.3 * 0.5
        confidence = 0.5
        shap_values = {}
    
    classification = score_to_classification(score)
    elapsed = time.time() - start
    
    return PredictionResult(
        gene=gene,
        variant=variant_str,
        meta_score=round(score, 4),
        classification=classification,
        confidence=round(confidence, 3),
        esm2_llr=features.get('esm2_llr'),
        esm2_wt_prob=features.get('esm2_wt_prob'),
        esm2_mut_prob=features.get('esm2_mut_prob'),
        esm2_entropy=features.get('esm2_entropy'),
        grantham_distance=features.get('grantham'),
        feature_importances={k: round(v, 4) for k, v in sorted(
            shap_values.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]} if shap_values else {},
        all_features={k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
    )


# ── API Endpoints ──

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE


@app.post("/api/predict", response_model=PredictionResult)
async def api_predict(query: VariantQuery):
    """Score a single missense variant."""
    if query.ref_aa not in AA_PROPS or query.alt_aa not in AA_PROPS:
        raise HTTPException(400, "Invalid amino acid. Use 1-letter codes (A, R, N, D, ...)")
    return predict_variant(query.gene, query.ref_aa, query.position, query.alt_aa)


@app.get("/api/predict/{gene}/{variant}")
async def api_predict_hgvs(gene: str, variant: str):
    """Score a variant by gene and protein change (e.g., /api/predict/BRCA1/C61G)."""
    # Parse variant string like C61G, R72P, etc.
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', variant.upper())
    if not m:
        # Try 3-letter: Cys61Gly
        m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', variant)
        if m:
            ref_aa = AA3TO1.get(m.group(1))
            alt_aa = AA3TO1.get(m.group(3))
            position = int(m.group(2))
        else:
            raise HTTPException(400, f"Cannot parse variant '{variant}'. Use format: C61G or Cys61Gly")
    else:
        ref_aa, position, alt_aa = m.group(1), int(m.group(2)), m.group(3)
    
    return predict_variant(gene.upper(), ref_aa, position, alt_aa)


@app.post("/api/batch", response_model=list[PredictionResult])
async def api_batch(query: BatchQuery):
    """Score a batch of variants."""
    results = []
    for v in query.variants[:100]:  # Limit to 100
        results.append(predict_variant(v.gene, v.ref_aa, v.position, v.alt_aa))
    return results


@app.get("/api/genes")
async def api_genes():
    """List genes with available protein sequences."""
    return {"genes": sorted(SEQUENCES.keys()), "count": len(SEQUENCES)}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "esm2_loaded": ESM2_SCORER is not None,
        "sequences_loaded": len(SEQUENCES),
    }


# ── HTML Template ──
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VUS Meta-Predictor 🧬</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f5f5; color: #333; }
        .header { background: linear-gradient(135deg, #1a237e, #4a148c);
                  color: white; padding: 2rem; text-align: center; }
        .header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        .header p { opacity: 0.8; }
        .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 12px; padding: 2rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; }
        .input-group { display: flex; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }
        .input-group label { display: block; font-weight: 600; margin-bottom: 0.25rem; font-size: 0.85rem; }
        .input-group input, .input-group select {
            padding: 0.6rem 0.8rem; border: 2px solid #e0e0e0; border-radius: 8px;
            font-size: 1rem; transition: border-color 0.2s; }
        .input-group input:focus { border-color: #4a148c; outline: none; }
        .field { flex: 1; min-width: 120px; }
        .field.gene { min-width: 150px; }
        .field.pos { max-width: 100px; }
        .field.aa { max-width: 80px; }
        button { background: linear-gradient(135deg, #4a148c, #1a237e); color: white;
                 border: none; padding: 0.8rem 2rem; border-radius: 8px; font-size: 1rem;
                 cursor: pointer; font-weight: 600; transition: transform 0.1s, opacity 0.2s; }
        button:hover { opacity: 0.9; transform: translateY(-1px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .quick-examples { margin-top: 1rem; }
        .quick-examples span { display: inline-block; background: #e8eaf6; padding: 0.3rem 0.7rem;
                                border-radius: 20px; margin: 0.2rem; cursor: pointer; font-size: 0.85rem; }
        .quick-examples span:hover { background: #c5cae9; }
        #result { display: none; }
        .score-display { text-align: center; padding: 1.5rem; }
        .score-number { font-size: 3.5rem; font-weight: 700; }
        .score-label { font-size: 1.3rem; font-weight: 600; margin-top: 0.5rem;
                       padding: 0.3rem 1rem; border-radius: 20px; display: inline-block; color: white; }
        .confidence { color: #666; margin-top: 0.5rem; }
        .features-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
        .feature-item { display: flex; justify-content: space-between; padding: 0.5rem;
                        background: #fafafa; border-radius: 6px; }
        .feature-name { color: #666; font-size: 0.85rem; }
        .feature-value { font-weight: 600; font-family: monospace; }
        .shap-bar { height: 6px; border-radius: 3px; margin-top: 4px; }
        .shap-positive { background: #ef5350; }
        .shap-negative { background: #43a047; }
        .esm2-section { margin-top: 1rem; padding: 1rem; background: #f3e5f5; border-radius: 8px; }
        .esm2-section h3 { color: #4a148c; margin-bottom: 0.5rem; }
        .loading { display: none; text-align: center; padding: 2rem; }
        .spinner { border: 3px solid #e0e0e0; border-top: 3px solid #4a148c;
                   border-radius: 50%; width: 40px; height: 40px;
                   animation: spin 0.8s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .api-docs { font-size: 0.9rem; color: #666; }
        .api-docs code { background: #f5f5f5; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.85rem; }
        footer { text-align: center; padding: 2rem; color: #999; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 VUS Meta-Predictor</h1>
        <p>Variant interpretation powered by ESM-2 protein language model + amino acid physicochemistry</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>Query a Missense Variant</h2>
            <div class="input-group" style="margin-top: 1rem;">
                <div class="field gene">
                    <label>Gene</label>
                    <input type="text" id="gene" placeholder="BRCA1" value="BRCA1">
                </div>
                <div class="field aa">
                    <label>Ref AA</label>
                    <input type="text" id="ref_aa" placeholder="C" value="C" maxlength="1" style="text-transform: uppercase;">
                </div>
                <div class="field pos">
                    <label>Position</label>
                    <input type="number" id="position" placeholder="61" value="61">
                </div>
                <div class="field aa">
                    <label>Alt AA</label>
                    <input type="text" id="alt_aa" placeholder="G" value="G" maxlength="1" style="text-transform: uppercase;">
                </div>
                <div class="field" style="display: flex; align-items: flex-end;">
                    <button onclick="predict()" id="predictBtn">Predict</button>
                </div>
            </div>
            <div class="quick-examples">
                <strong>Try:</strong>
                <span onclick="fillExample('BRCA1','C','61','G')">BRCA1 C61G</span>
                <span onclick="fillExample('TP53','R','248','W')">TP53 R248W</span>
                <span onclick="fillExample('TP53','R','72','P')">TP53 R72P</span>
                <span onclick="fillExample('MYH7','R','403','Q')">MYH7 R403Q</span>
                <span onclick="fillExample('LDLR','C','302','G')">LDLR C302G</span>
                <span onclick="fillExample('SCN1A','R','1648','C')">SCN1A R1648C</span>
                <span onclick="fillExample('CFTR','G','551','D')">CFTR G551D</span>
                <span onclick="fillExample('PTEN','R','130','Q')">PTEN R130Q</span>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 1rem; color: #666;">Running ESM-2 + meta-learner...</p>
        </div>
        
        <div id="result">
            <div class="card">
                <div class="score-display">
                    <div id="variantTitle" style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;"></div>
                    <div class="score-number" id="scoreValue"></div>
                    <div class="score-label" id="classLabel"></div>
                    <div class="confidence" id="confValue"></div>
                </div>
            </div>
            
            <div class="card esm2-section" id="esm2Card">
                <h3>🤖 ESM-2 Protein Language Model</h3>
                <div class="features-grid" id="esm2Features"></div>
            </div>
            
            <div class="card">
                <h3>🔬 Amino Acid Properties</h3>
                <div class="features-grid" id="aaFeatures"></div>
            </div>
            
            <div class="card" id="shapCard" style="display:none;">
                <h3>📊 Feature Contributions (SHAP)</h3>
                <div id="shapFeatures"></div>
            </div>
        </div>
        
        <div class="card api-docs">
            <h3>API Endpoints</h3>
            <p style="margin-top: 0.5rem;">
                <code>GET /api/predict/{gene}/{variant}</code> — e.g., <code>/api/predict/BRCA1/C61G</code><br>
                <code>POST /api/predict</code> — JSON body: <code>{"gene": "BRCA1", "ref_aa": "C", "position": 61, "alt_aa": "G"}</code><br>
                <code>POST /api/batch</code> — Score up to 100 variants at once<br>
                <code>GET /api/genes</code> — List available genes<br>
                <code>GET /api/health</code> — Health check
            </p>
        </div>
    </div>
    
    <footer>
        VUS Meta-Predictor v1.0 · ESM-2 + XGBoost + Isotonic Calibration ·
        <a href="https://github.com/raonyguimaraes/vus-meta-predictor" style="color: #666;">GitHub</a>
    </footer>
    
    <script>
    function fillExample(gene, ref, pos, alt) {
        document.getElementById('gene').value = gene;
        document.getElementById('ref_aa').value = ref;
        document.getElementById('position').value = pos;
        document.getElementById('alt_aa').value = alt;
        predict();
    }
    
    async function predict() {
        const gene = document.getElementById('gene').value.toUpperCase();
        const ref_aa = document.getElementById('ref_aa').value.toUpperCase();
        const position = parseInt(document.getElementById('position').value);
        const alt_aa = document.getElementById('alt_aa').value.toUpperCase();
        
        if (!gene || !ref_aa || !position || !alt_aa) { alert('Fill all fields'); return; }
        
        document.getElementById('result').style.display = 'none';
        document.getElementById('loading').style.display = 'block';
        document.getElementById('predictBtn').disabled = true;
        
        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({gene, ref_aa, position, alt_aa})
            });
            const data = await resp.json();
            
            document.getElementById('variantTitle').textContent = `${gene} ${data.variant}`;
            document.getElementById('scoreValue').textContent = data.meta_score.toFixed(4);
            
            const label = document.getElementById('classLabel');
            label.textContent = data.classification;
            const colors = {
                'Pathogenic': '#d32f2f', 'Likely Pathogenic': '#f57c00',
                'Uncertain Significance': '#757575',
                'Likely Benign': '#43a047', 'Benign': '#1b5e20'
            };
            label.style.background = colors[data.classification] || '#757575';
            document.getElementById('scoreValue').style.color = colors[data.classification] || '#333';
            
            document.getElementById('confValue').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            
            // ESM-2 features
            const esm2Div = document.getElementById('esm2Features');
            esm2Div.innerHTML = '';
            const esm2Items = [
                ['Log-Likelihood Ratio', data.esm2_llr],
                ['WT Probability', data.esm2_wt_prob],
                ['Mutant Probability', data.esm2_mut_prob],
                ['Position Entropy', data.esm2_entropy],
            ];
            esm2Items.forEach(([name, val]) => {
                if (val !== null && val !== undefined) {
                    esm2Div.innerHTML += `<div class="feature-item"><span class="feature-name">${name}</span><span class="feature-value">${val.toFixed(4)}</span></div>`;
                }
            });
            
            // AA features
            const aaDiv = document.getElementById('aaFeatures');
            aaDiv.innerHTML = '';
            const aaItems = [
                ['Grantham Distance', data.grantham_distance],
                ['Hydrophobicity Δ', data.all_features?.hydro_change],
                ['Volume Δ', data.all_features?.volume_change],
                ['Polarity Δ', data.all_features?.polarity_abs_change],
                ['To Proline', data.all_features?.to_proline],
                ['From Cysteine', data.all_features?.from_cysteine],
            ];
            aaItems.forEach(([name, val]) => {
                if (val !== null && val !== undefined) {
                    aaDiv.innerHTML += `<div class="feature-item"><span class="feature-name">${name}</span><span class="feature-value">${typeof val === 'number' ? val.toFixed(3) : val}</span></div>`;
                }
            });
            
            // SHAP
            const shapDiv = document.getElementById('shapFeatures');
            const shapCard = document.getElementById('shapCard');
            if (data.feature_importances && Object.keys(data.feature_importances).length > 0) {
                shapCard.style.display = 'block';
                shapDiv.innerHTML = '';
                const maxShap = Math.max(...Object.values(data.feature_importances).map(Math.abs));
                Object.entries(data.feature_importances).forEach(([name, val]) => {
                    const pct = Math.abs(val) / maxShap * 100;
                    const cls = val > 0 ? 'shap-positive' : 'shap-negative';
                    const dir = val > 0 ? '→ pathogenic' : '→ benign';
                    shapDiv.innerHTML += `
                        <div style="margin-bottom: 0.5rem;">
                            <div style="display:flex; justify-content:space-between;">
                                <span class="feature-name">${name}</span>
                                <span class="feature-value" style="color: ${val > 0 ? '#d32f2f' : '#43a047'}">${val > 0 ? '+' : ''}${val.toFixed(4)} ${dir}</span>
                            </div>
                            <div class="shap-bar ${cls}" style="width: ${pct}%"></div>
                        </div>`;
                });
            } else {
                shapCard.style.display = 'none';
            }
            
            document.getElementById('result').style.display = 'block';
        } catch (e) {
            alert('Error: ' + e.message);
        } finally {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('predictBtn').disabled = false;
        }
    }
    </script>
</body>
</html>
"""
