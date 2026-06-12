# Data sources & provenance

Raw inputs total several GB, so this repo ships small **processed** tables in full and **samples** of large raw files. Everything can be reproduced from the original sources below.

## data/processed/ (full files — outputs of `02_data_cleaning.ipynb`)

| File | Contents |
|------|----------|
| `global_dc_counts_tidy.parquet` | Country-level data-centre counts scraped from Cloudscene (Aug 2025) |
| `world_pop_long.parquet`, `world_gdp_long.parquet`, `world_pop_gdp_merged.parquet` | World Bank population & GDP, long format |
| `seds_all.parquet`, `complete_tidy.parquet` | US EIA SEDS state energy series, 1930–2023 |
| `consumption_btu_tidy.parquet`, `consumption_phy_tidy.parquet` | SEDS consumption (BTU / physical units) |
| `production_tidy.parquet`, `prices_tidy.parquet`, `indicators_tidy.parquet`, `co2_tidy.parquet` | SEDS production, prices, indicators, CO₂ |
| `ukpn_timeseries_tidy.parquet` | UKPN data-centre half-hourly demand profiles, tidy |
| `gdelt_sampled_filtered.csv` | GDELT GKG records filtered for AI/cloud/data-centre coverage (2015–2025 sample) |
| `gdelt_ai_cloud_datacenters_sentiment.parquet` | Tone/sentiment series derived from GDELT |
| `sentiment140_filtered.csv` | Filtered Sentiment140 tweets used for local sentiment modelling |

## data/samples/

| File | Note |
|------|------|
| `workstation_2021_*_tidy_SAMPLE.parquet` | ~150k-row systematic samples of the 2021 server telemetry tidy tables (full: 2.9M + 1.5M rows) |
| `workstation_*_raw_SAMPLE.csv` | First 5,000 rows of the raw telemetry CSVs (full: 1.2GB + 532MB) |
| `Complete_SEDS_SAMPLE.csv` | First 5,000 rows of full SEDS extract (76MB) |
| `ukpn-data-centre-demand-profiles_SAMPLE.csv` | First 5,000 rows (full: 126MB) |
| `World_bank_population.csv`, `co2_all.csv`, `energy_indicators.csv`, `prod_data.xlsx`, `ukpn-data-centre-utilisation.csv`, `data-centres-worldwide-general-dataset.csv` | Small raw files, included in full |

## Original sources

| Source | Used for | Link |
|--------|----------|------|
| US EIA State Energy Data System (SEDS), 1930–2023 | US energy consumption/production/prices/CO₂ | https://www.eia.gov/state/seds/ |
| World Bank — Population (SP.POP.TOTL) & GDP | Per-capita normalisation | https://data.worldbank.org/indicator/SP.POP.TOTL |
| UK Power Networks Open Data — data-centre demand profiles & utilisation | UK demand reconstruction | https://ukpowernetworks.opendatasoft.com/ |
| Cloudscene | Country-level data-centre counts (scraped Aug 2025; counts reflect that snapshot) | https://cloudscene.com/ |
| GDELT GKG v2 | News discourse & tone, 2015–2025 | http://data.gdeltproject.org/gdeltv2/ |
| Sentiment140 | Tweet sentiment training corpus | http://help.sentiment140.com/for-students |
| Server energy-consumption telemetry (2021 workstation logs, May–Dec) | Surrogate model training | Public research dataset; see dissertation §Methodology for citation |

**Note on scraping:** the Cloudscene scraper was run once (Aug 2025) for academic research; respect the site's terms of service before re-running.
