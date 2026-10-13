# Campaign definitions

One file per campaign, sourced by `eval/campaign.sh`. `TEMPLATE.env` documents every variable. The run names these
definitions produce are the names of the published runs, so a re-run lands next to them, and the results tree and the
reports line up.

| definition | checkpoint(s) | settings | published result (3 seeds, SR / score, successes of 6300) |
|---|---|---|---|
| `pantheon33k_latent_nas24.env` | Pantheon33k sana latent (three-view strip) | n_action_steps 24 | **16.29 / 22.84, 1052** (best on record) |
| `pantheon33k_latent.env` | the same checkpoint | whole chunks | 14.78 / 21.29, 958 |
| `pantheon33k_latent_videoaug.env` | Pantheon33k sana latent trained with video augmentation | whole chunks | 927 |
| `pantheon33k_latent_videoaug_nas24.env` | the same checkpoint | n_action_steps 24 | 922 |
| `pantheon14k_pixel.env` | Pantheon14k sana pixel (320x512 canvas) | whole chunks | 14.87 / 21.09, 956 |
| `pantheon14k_pixel_nas24.env` | the same checkpoint | n_action_steps 24 | 15.39 / 21.95, 984 |
| `pantheon14k_pixel_videoaug.env` | Pantheon14k sana pixel trained with video augmentation | whole chunks | 13.15 / 19.82, 852 |
| `pantheon14k_nocfg.env` | Pantheon14k sana pixel trained without condition dropout | guidance off (video CFG 1, action CFG 1) | 15.23 / 21.76, 979 |
| `pantheon14k_openwam.env` | Pantheon14k OpenWAM canvas (384x320) | whole chunks | in progress (2026-10-13) |
| `pantheon2k_latent_pair.env` | the two Pantheon2k sana latents (fixed resize, video augmentation), in parallel | whole chunks | 14.31 / 20.99, 929; 15.46 / 21.62, 1018 |

Everything not listed in a definition follows the defaults of `eval/run.sh`: 10 steps, video CFG 6 / action CFG 1, flow
shifts 5 / 1, end-effector actions, and the adapter's own view resize. `CAMPAIGN` keeps the chain-directory name of the
original run, so `status` and `combined` find the existing runs.
