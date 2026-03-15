# Experiment Summary

- Encoder: `mobilevitv2_150.cvnets_in1k`
- Decoder: `plain`
- Input mode: `central_roi`
- Train ROI: `0.75`
- Image size: `96`
- Bottleneck: `32`
- Score mode: `pixel_quantile`
- Score param: `0.8000`
- Anomaly score: `reconstruction`
- Threshold: `0.00205390`
- Best epoch: `6`
- Best val loss: `0.000565`

## Test Metrics

- Accuracy: `0.9315`
- TPR: `0.7829`
- TNR: `0.9367`

## Files

- `metrics.json`
- `loss_curves.png`
- `score_histogram.png`
- `latent_space.png`
- `mobilevitv2_150_cvnets_in1k_autoencoder.pt`
- `reconstructions/`
