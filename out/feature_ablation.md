# Feature ablation

Baseline (all 14 features): MAE = 0.146, accept = 0.58, accepted CER = 0.043, bad accepts = 0.12.

| Removed | Kind | MAE | ΔMAE | Accept | Accepted CER |
|---|---|---|---|---|---|
| [group] sharpness | group | 0.213 | +0.068 | 0.53 | 0.1125 |
| text_height | single | 0.180 | +0.035 | 0.57 | 0.0475 |
| tilt_deg | single | 0.155 | +0.009 | 0.42 | 0.0472 |
| [group] geometry | group | 0.155 | +0.009 | 0.42 | 0.0472 |
| tenengrad | single | 0.149 | +0.003 | 0.57 | 0.0434 |
| hf_ratio | single | 0.147 | +0.002 | 0.57 | 0.0434 |
| illum_gradient | single | 0.145 | -0.000 | 0.57 | 0.0434 |
| [group] illumination | group | 0.145 | -0.001 | 0.54 | 0.0452 |
| bright_frac | single | 0.145 | -0.001 | 0.57 | 0.0434 |
| glare_blob_frac | single | 0.145 | -0.001 | 0.57 | 0.0434 |
| glare_contrast | single | 0.145 | -0.001 | 0.57 | 0.0434 |
| blockiness | single | 0.144 | -0.001 | 0.60 | 0.0436 |
| contrast | single | 0.144 | -0.001 | 0.57 | 0.0434 |
| lap_var | single | 0.144 | -0.002 | 0.58 | 0.0425 |
| mean | single | 0.141 | -0.005 | 0.60 | 0.0417 |
| std | single | 0.140 | -0.006 | 0.60 | 0.044 |
| noise_sigma | single | 0.140 | -0.006 | 0.58 | 0.0425 |
| [group] noise | group | 0.128 | -0.018 | 0.61 | 0.0431 |

Largest increases:
- [group] sharpness: +0.068 MAE
- text_height: +0.035 MAE
- tilt_deg: +0.009 MAE