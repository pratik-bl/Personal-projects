-- Country-level data-centre counts, typed and de-duplicated
select
    country,
    cast(dc_count as integer) as dc_count
from {{ source('processed', 'global_dc_counts_tidy') }}
where dc_count is not null
