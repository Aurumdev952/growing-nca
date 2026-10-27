# Optimisation

## Training throughput

All timings are from an RTX 5090 Laptop GPU with a 95 W power cap.

| setup | time per iteration (batch 8) |
|---|---|
| eager PyTorch | 51.6 ms |
| `torch.compile` on the step | 32.1 ms |
| compiled plus bf16 autocast | 31.3 ms |

Compiling the step makes training 1.6 times faster (one run of 4 repetitions). bf16 adds almost nothing, so training stays in fp32.

Iteration time grows linearly with batch size: 49 ms at batch 8, 226 ms at 32, and 528 ms at 64. The GPU stays at its power cap, with clocks throttled to about 1.4 GHz. Running more processes in parallel therefore gains nothing. With three running side by side, each model takes about 15 minutes.

## Model quality

Each model is scored on MSE at step 2000 plus mean MSE 200 steps after five fixed damage patterns. Seed-to-seed variance turned out larger than the effect of any training change tried.

| experiment | runs | finding |
|---|---|---|
| overflow penalty, gecko | 3 seeds vs 3 | Regeneration unchanged (median 0.00046 vs 0.00039, with overlapping ranges). Rotated growth better for all three seeds (0.0011 to 0.0020 vs 0.0022 to 0.0034). Suggestive, but only three seeds. |
| overflow penalty, spiderweb | 3 vs 3 | Never blows up (step-2000 MSE 0.0005 to 0.0009, against a baseline range of 0.0001 to 0.0143). The best single model is still a baseline seed. |
| batch 16, spiderweb | 3 vs 3 | Ranges overlap with batch 8. Seed 1 blows up at both batch sizes. Inconclusive. |
| best of 3 seeds | 3 seeds per emoji | Checked on 32 held-out rollouts with a different RNG seed. Gecko improved from 0.00062 to 0.00034, spiderweb from 0.00935 to 0.00092, and tree from 0.00411 to 0.00178. Fish, ladybug and pretzel did not improve. |

The shipped models are the best-of-seeds winners. Filmstrips and tables for every candidate are in [`eval/hillclimb/`](../eval/hillclimb/).

## Viewer

[viewer.md](viewer.md#benchmarking) has the viewer's speed. At the default grid size it runs about ten times faster than smooth playback needs.
