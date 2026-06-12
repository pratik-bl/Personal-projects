-- Hour-of-day power profile of the monitored workstation
{{ config(
    materialized='external',
    location='../outputs/marts/workstation_diurnal_power.csv',
    format='csv'
) }}

select
    extract(hour from datetime)         as hour_of_day,
    count(*)                            as n_readings,
    round(avg(power_w), 2)              as mean_power_w,
    round(quantile_cont(power_w, 0.95), 2) as p95_power_w,
    round(avg(cpu_util_pct), 2)         as mean_cpu_util_pct,
    round(avg(gpu_util_pct), 2)         as mean_gpu_util_pct
from {{ ref('stg_workstation') }}
group by 1
order by 1
