-- Sampled workstation power telemetry (2021 May-Aug)
select
    datetime,
    power_w,
    cpu_util_pct,
    gpu_util_pct,
    ram_util_pct,
    cpu_temp_c,
    gpu_temp_c
from {{ source('samples', 'workstation_2021_may_aug_tidy_SAMPLE') }}
where power_w is not null
