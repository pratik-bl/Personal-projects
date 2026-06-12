-- World Bank population, long format
select
    country,
    iso3,
    cast(year as integer) as year,
    cast(population as bigint) as population
from {{ source('processed', 'world_pop_long') }}
