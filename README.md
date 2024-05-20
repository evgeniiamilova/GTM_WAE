<div align="center">
    <h3><i>GTM-guided WAE for Rational Design of Antimicrobial and Anti-biofilm Peptides</i></h3>

---

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/yourproject/license)

</div>

## Overview

Our project presents a computational pipeline for the rational de novo design of antimicrobial and anti-biofilm peptides based on an explainable artificial intelligence (XAI) framework.
- 🧬 Combines Wasserstein Autoencoder (WAE) and Generative Topographic Mapping (GTM) for rational peptide design.
- 📊 WAE learns the latent representation of the peptide space.
- 🗺️ GTM provides a 2D illustrative depiction of the latent space for generating novel peptides.
- 🔬 Experimentally verified efficacy against MRSA with a 100% hit rate in targeting biofilms.
- 🚀 Optimizable for additional peptide properties like cytotoxicity, aggregation, and proteolytic stability.

## As for GTM, refer to :
[![Lab Website](https://img.shields.io/badge/Website-Laboratory%20of%20Chemoinformatics-blue)](http://complex-matter.unistra.fr/en/research-teams/laboratory-of-chemoinformatics/software-development/#c88715)

## Data set
The standardized pre-processed data set collected from public data bases used for training of GTM_WAE (in-house peptides are not included):
* 44392 linear peptides of lengths 10-14 (inclusive) from TrEMBL, DBAASP, SATPdb, SwissProt, FermFooDb, Hemolytik, NeuroPedia, BaAMPs and APD3 [https://huggingface.co/datasets/karinapikalyova/peptides/tree/main]
## Quick Start

To set up environment for GTM_WAE, run:

```bash
conda env create -f GTM_WAE.yml
conda activate GTM_WAE

pip install "GTM_WAE[all]"
```

## Main developers
- **Karina Pikalyova** - karinapikalyova@gmail.com
- **Dr. Tagir Akhmetshin** - tagirshin@gmail.com
- **Dr. Alexey Orlov** - 2alexeyorlov@gmail.com
- **Prof. Alexandre Varnek** - varnek@unistra.fr


