# Historical campaign drivers

The per-recipe drivers (`<recipe>.sh`: one run of one checkpoint) and chains (`*_seeds.sh`, `*_lanes.sh`: several seeds or
lanes) that produced the published runs between 2026-09-10 and 2026-10-13, kept verbatim as the record of how each run
was launched. They are hard-wired to the reference site (user, host names, paths, port ranges). Do not run them
elsewhere.

New campaigns use `../run.sh` (the per-run driver for every adapter) and `../campaign.sh` with a definition in `../defs/`.
Those definitions reproduce the run names, master arguments, adapter overrides, smoke rules and summaries of the chains
below:

| chain here | definition |
|---|---|
| `pantheon14k_openwam_allgpu_seeds.sh` | `../defs/pantheon14k_openwam.env` |
| `pantheon14k_nocfg_allgpu_seeds.sh` | `../defs/pantheon14k_nocfg.env` |
| `pantheon14k_pixel_allgpu_seeds.sh` | `../defs/pantheon14k_pixel.env` |
| `pantheon14k_nas24_allgpu_seeds.sh` | `../defs/pantheon14k_pixel_nas24.env` |
| `pantheon14k_videoaug_allgpu_seeds.sh` | `../defs/pantheon14k_pixel_videoaug.env` |
| `pantheon33k_latent_allgpu_seeds.sh` | `../defs/pantheon33k_latent.env` |
| `pantheon33k_latent_nas24_allgpu_seeds.sh` | `../defs/pantheon33k_latent_nas24.env` |
| `pantheon33k_videoaug_allgpu_seeds.sh` | `../defs/pantheon33k_latent_videoaug.env` |
| `pantheon33k_videoaug_nas24_allgpu_seeds.sh` | `../defs/pantheon33k_latent_videoaug_nas24.env` |
| `pantheon_stretch_pair_seeds.sh` | `../defs/pantheon2k_latent_pair.env` |

The per-run driver they all call, `vanilla34k_eefabs_f33fps8.sh` (SANA_WAM), corresponds to `../run.sh` with
`ADAPTER=SANA_WAM`. `mot_eefabs_sana_pixel_f33fps8.sh` corresponds to `ADAPTER=SANA_MOT`, and
`causal_p14k_eefabs_f33fps8.sh` to `ADAPTER=SANA_WAM_CAUSAL`.
