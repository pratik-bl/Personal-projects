-- UKPN half-hourly data-centre utilisation (0-1)
select
    site_id,
    site_type,
    voltage_level,
    datetime_local,
    utilisation
from {{ source('processed', 'ukpn_timeseries_tidy') }}
where utilisation is not null
