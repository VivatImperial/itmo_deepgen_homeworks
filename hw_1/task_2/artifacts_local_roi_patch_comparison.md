# Local ROI/Patch Comparison

Сравнение локальных экспериментов относительно основного full-image run на `mobilevitv2_150.cvnets_in1k`.

## Full-image

- `TPR=0.9535`
- `TNR=0.9179`
- `accuracy=0.9191`

## Central ROI

- `train_roi=0.75`
- `TPR=0.7829`
- `TNR=0.9367`
- `accuracy=0.9315`

## Random Patch

- `train_patch_size=64`
- `tile_size=64`
- `tile_stride=16`
- `tile_agg_mode=max`
- `TPR=0.7907`
- `TNR=0.9045`
- `accuracy=0.9006`

## Вывод

Локальный трек не улучшил минимум между `TPR` и `TNR`, поэтому основным остается `artifacts_mobilevitv2_150_96_b32_wd5e4`.
