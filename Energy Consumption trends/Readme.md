# Energy Generation and Consumption Trends

A quick, clear analysis of how energy is generated and used across countries, comparing developed and developing groups. Runs from a single Jupyter Notebook.

## What this does
- Loads country-year energy data
- Cleans and joins tables
- Builds comparable metrics (renewable share, fossil share, CO₂ per kWh, energy per capita, access to electricity)
- Saves a few figures and small summary tables

## Files
- `Energy Consumption trends/energy_generation_consumption.ipynb`  ← open and run
- `data/sample/`                                    ← tiny CSVs so the notebook runs fast
- `reports/figures/`                                ← charts are saved here
- `artifacts/`                                      ← summary CSVs

## Data sources
List the sources you used and links here, for example:
- Our World in Data energy datasets
- World Bank indicators
Add any licence notes in `data/README.md`.

## Quick start
```bash
# Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
