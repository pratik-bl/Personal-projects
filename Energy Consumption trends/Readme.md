# Energy Generation and Consumption Trends

Quick notebook exploring how energy is generated and used across countries, with a simple comparison between developed and developing groups.

## What this notebook does
- Loads a country–year energy dataset (Our World in Data).
- Cleans and filters to real countries using ISO3 codes.
- Groups by **Category** (developed vs developing) and **Region**.
- Plots median trends over time for:
  - Renewables share
  - Greenhouse gas emissions
  - Primary energy consumption
- Shows a 2021 snapshot of **Energy per Capita** by country.
- Aggregates regional totals for **Solar**, **Hydro**, and **Wind** electricity.

## How to run
Create a virtual environment (Python 3.11 recommended)  
`pip install -r requirements.txt`  
Open and run: `energy_generation_consumption.ipynb`

## Data
This project uses the dataset from the following repository:

Original dataset: https://github.com/owid/energy-data/tree/master

I did not include the full dataset in this repo. Please download it from the source above and place the CSV(s) here: data/raw/owid-energy-data.csv
