# Experiment Summary

- Encoder: `mobilevitv2_150.cvnets_in1k`
- Decoder: `plain`
- Input mode: `random_patch`
- Train patch size: `64`
- Tile size: `64`
- Tile stride: `16`
- Tile aggregation: `max`
- Image size: `96`
- Bottleneck: `32`
- Score mode: `pixel_quantile`
- Score param: `0.8000`
- Anomaly score: `reconstruction`
- Threshold: `0.00546212`
- Best epoch: `6`
- Best val loss: `0.001404`

## Test Metrics

- Accuracy: `0.9006`
- TPR: `0.7907`
- TNR: `0.9045`

## Files

- `metrics.json`
- `loss_curves.png`
- `score_histogram.png`
- `latent_space.png`
- `mobilevitv2_150_cvnets_in1k_autoencoder.pt`
- `reconstructions/`
