# Prepared data dictionary

| Field | Meaning | Use |
|---|---|---|
| `event_id` | Stable source-qualified event identifier | joins and API evidence |
| `timestamp_utc` | Normalized observation time in UTC | chronology and split |
| `latitude`, `longitude` | Event coordinates in WGS84 degrees | mapping and model features |
| `frp` | Fire radiative power | severity feature |
| `brightness_temperature` | Source thermal brightness temperature | thermal feature |
| `frp_uncertainty` | Source FRP uncertainty | quality feature |
| `prior_detections_7d/30d/90d` | Strictly historical detections | persistence features |
| `stationary_count_90d` | Historical nearby detections consistent with stationary heat | persistence feature |
| `nearest_flare_distance_m` | Distance to nearest known gas-flaring point | industrial context |
| `nearest_industrial_distance_m` | Distance to nearest industrial facility | industrial context |
| `label_stage_1` | Emergency/transient vs persistent/non-emergency candidate | Stage 1 target |
| `label_stage_2` | Industrial-fire vs persistent industrial heat/flare candidate | Stage 2 target |
| `supervised_eligible` | Conservative label-quality gate | training filter |
| `split` | `development` or untouched `ranking` partition | leakage control |

Rows with uncertain or conflicting evidence remain visible for review and are
not silently converted into supervised labels.
