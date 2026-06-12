# dbt + DuckDB transformation layer

Rebuilds the analytics tables as tested, documented dbt models running on
DuckDB — no server or paid service needed.

```bash
pip install dbt-duckdb
cd dbt
DBT_PROFILES_DIR=. dbt build     # PowerShell: $env:DBT_PROFILES_DIR="."; dbt build
```

`dbt build` reads the parquet/CSV sources straight from `../data/`, builds
5 staging views and 4 marts, runs 18 schema tests, and exports each mart as
CSV to `../outputs/marts/` for BI tools (Power BI, Tableau, Excel).

| Mart | Grain | Feeds |
|------|-------|-------|
| `mart_dc_density_by_country` | country | world density map / bar charts |
| `mart_ukpn_monthly_utilisation` | month x site type | utilisation trend lines |
| `mart_sentiment_yearly` | year | news-tone trend (2015-2025) |
| `mart_workstation_diurnal_power` | hour of day | power profile chart |
