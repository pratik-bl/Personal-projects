-- Data centres per million population (2024), the dissertation's density lens
{{ config(
    materialized='external',
    location='../outputs/marts/dc_density_by_country.csv',
    format='csv'
) }}

with pop_2024 as (
    select country, population
    from {{ ref('stg_population') }}
    where year = 2024
)

select
    dc.country,
    dc.dc_count,
    p.population,
    round(dc.dc_count * 1e6 / p.population, 3) as dc_per_million
from {{ ref('stg_dc_counts') }} dc
join pop_2024 p
  on trim(lower(dc.country)) = trim(lower(p.country))
order by dc_per_million desc
