-- Monthly mean and p95 utilisation per UKPN site type
{{ config(
    materialized='external',
    location='../outputs/marts/ukpn_monthly_utilisation.csv',
    format='csv'
) }}

select
    date_trunc('month', datetime_local) as month,
    site_type,
    count(distinct site_id)             as n_sites,
    round(avg(utilisation), 4)          as mean_utilisation,
    round(quantile_cont(utilisation, 0.95), 4) as p95_utilisation
from {{ ref('stg_ukpn_timeseries') }}
group by 1, 2
order by 1, 2
