-- GDELT article tone scores for AI / cloud / data-centre coverage
select
    cast(year as integer) as year,
    cast(tone as double)  as tone,
    sentiment
from {{ source('processed', 'gdelt_sampled_filtered') }}
where tone is not null
