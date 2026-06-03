# PhysTrans

PhysTrans is a physics-aware transferable framework for cold-start photovoltaic (PV) forecasting.
It integrates a clear-sky physical baseline with learnable residual correction, and optionally fuses
satellite cloud-image features for improved cross-site transfer performance.

## Features
- Physics-informed PV forecasting (clear-sky baseline + residual correction)
- Cross-site transfer learning (zero-shot transfer / limited-data adaptation)
- Optional cloud-image fusion with learnable gating and residual projection
- Support for long-term forecasting experiments

## Requirements
- Python >= 3.8
- PyTorch (GPU recommended)

Install dependencies:
```bash
pip install -r PhysTrans/PhysTrans/docs/requirements.txt
