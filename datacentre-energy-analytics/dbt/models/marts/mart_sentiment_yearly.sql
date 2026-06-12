-- Yearly news tone for AI / cloud / data-centre coverage (GDELT, 2015-2025)
{{ config(
    materialized='external',
    location='../outputs/marts/sentiment_yearly.csv',
    format='csv'
) }}

select
    year,
    count(*)                                                  as n_articles,
    round(avg(tone), 3)                                       as mean_tone,
    round(avg(case when sentiment = 'positive' then 1.0 else 0.0 end), 3) as share_positive,
    round(avg(case when sentiment = 'negative' then 1.0 else 0.0 end), 3) as share_negative
from {{ ref('stg_gdelt_sentiment') }}
group by 1
order by 1
