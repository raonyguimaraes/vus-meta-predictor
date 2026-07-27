## Tweet Options

### Option 1: The hook (short, punchy)

🧬 Built a complete variant interpretation pipeline in one session:

• ESM-2 protein language model
• AlphaMissense integration  
• ACMG evidence automation
• Phenotype-aware ranking (HPO)
• Clinical report generation

Humbling finding: AlphaMissense alone (AUROC 0.961) beats my full ensemble (0.932).

But the real value? Automated ACMG classification + phenotype matching — the stuff no single score gives you.

8,774 lines of Python. Open source.

github.com/raonyguimaraes/vus-meta-predictor

---

### Option 2: The lesson learned (more personal)

I built a VUS interpretation pipeline with ESM-2, AlphaMissense, conservation scores, and an XGBoost meta-learner.

Then I benchmarked it properly.

AlphaMissense standalone: AUROC 0.961
My fancy ensemble: AUROC 0.932

🫠

The lesson: don't build an ensemble to beat a foundation model on its own task.

Instead, build what it CAN'T do:
→ Non-missense variants (80% of pathogenic ClinVar)
→ Phenotype-aware ranking
→ Automated ACMG evidence chains
→ Clinical reports

That's where the gap is.

github.com/raonyguimaraes/vus-meta-predictor

---

### Option 3: The visual (for LinkedIn/threads)

🧬 VUS Meta-Predictor — open source clinical variant interpretation

What it does:
  Patient phenotype (seizures, ID)
  + Candidate variants from exome
  → Phenotype-aware ranking
  → ACMG classification with cited evidence  
  → Clinical report

Real results (temporal hold-out, 123K test variants):
  📊 AUROC: 0.932
  ✅ NPV: 97.3% (benign calls are reliable)
  🏥 Automated ACMG + phenotype matching

Stack: ESM-2 · AlphaMissense · ClinVar · HPO · XGBoost · FastAPI

github.com/raonyguimaraes/vus-meta-predictor

#bioinformatics #genomics #clinicalgenomics #machinelearning #opensource
