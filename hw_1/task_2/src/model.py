from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from timm.data import resolve_model_data_config
from torch import nn
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV3Autoencoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "mobilevitv2_150.cvnets_in1k",
        encoder_out_index: int = 4,
        bottleneck_channels: int = 128,
        image_size: int = 64,
        freeze_encoder: bool = False,
        decoder_mode: str = "plain",
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.image_size = image_size
        self.freeze_encoder = freeze_encoder
        self.decoder_mode = decoder_mode
        self.skip_channels: list[int] = []
        if encoder_name.startswith("facebook/vit-mae"):
            self.encoder_mode = "hf_vit_mae"
            processor = AutoImageProcessor.from_pretrained(encoder_name, use_fast=True)
            config = ViTMAEConfig.from_pretrained(encoder_name)
            config.mask_ratio = 0.0
            self.encoder = ViTMAEModel.from_pretrained(encoder_name, config=config)
            encoder_channels = int(config.hidden_size)
            mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        elif ".mae" in encoder_name and encoder_name.startswith("vit_"):
            self.encoder_mode = "tokens"
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=True,
                num_classes=0,
                img_size=image_size,
            )
            encoder_channels = int(self.encoder.num_features)
            data_config = resolve_model_data_config(self.encoder)
            mean = torch.tensor(data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(data_config["std"], dtype=torch.float32).view(1, 3, 1, 1)
        else:
            self.encoder_mode = "features"
            try:
                self.encoder = timm.create_model(
                    encoder_name,
                    pretrained=True,
                    features_only=True,
                    out_indices=tuple(range(encoder_out_index + 1)),
                    img_size=image_size,
                )
            except TypeError:
                self.encoder = timm.create_model(
                    encoder_name,
                    pretrained=True,
                    features_only=True,
                    out_indices=tuple(range(encoder_out_index + 1)),
                )
            feature_channels = list(self.encoder.feature_info.channels())
            encoder_channels = feature_channels[-1]
            self.skip_channels = feature_channels[:-1]
            data_config = resolve_model_data_config(self.encoder)
            mean = torch.tensor(data_config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(data_config["std"], dtype=torch.float32).view(1, 3, 1, 1)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Hardswish(),
            nn.Dropout2d(p=0.1),
        )

        if self.decoder_mode == "skip" and self.encoder_mode != "features":
            raise ValueError("decoder_mode='skip' is only supported for feature-map encoders.")

        if self.decoder_mode == "skip":
            self.skip_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(skip_channels, bottleneck_channels, kernel_size=1),
                        nn.BatchNorm2d(bottleneck_channels),
                    )
                    for skip_channels in reversed(self.skip_channels)
                ]
            )
            self.skip_refine_blocks = nn.ModuleList(
                [ConvBlock(bottleneck_channels, bottleneck_channels) for _ in self.skip_channels]
            )
            decoder_channels = [128, 64, 32]
        else:
            self.skip_projections = nn.ModuleList()
            self.skip_refine_blocks = nn.ModuleList()
            decoder_channels = [256, 128, 64, 32, 16]

        self.decoder_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for next_channels in decoder_channels:
            self.decoder_blocks.append(ConvBlock(current_channels, next_channels))
            current_channels = next_channels
        self.output_head = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_channels),
            nn.Hardswish(),
            nn.Conv2d(current_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _skip_features = self.encode_features(x)
        return encoded

    def encode_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        normalized = (x - self.input_mean) / self.input_std
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_outputs = self._forward_encoder(normalized)
        else:
            encoder_outputs = self._forward_encoder(normalized)

        if self.encoder_mode == "features":
            feature_pyramid = [feature.contiguous() for feature in encoder_outputs]
            deepest_feature = feature_pyramid[-1]
            skip_features = feature_pyramid[:-1]
        else:
            deepest_feature = encoder_outputs
            skip_features = []
        return self.bottleneck(deepest_feature), skip_features

    def _forward_encoder(self, normalized: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        if self.encoder_mode == "hf_vit_mae":
            encoded = self.encoder(pixel_values=normalized.contiguous()).last_hidden_state
            if self._has_cls_token(encoded.shape[1]):
                encoded = encoded[:, 1:, :]
            batch_size, token_count, channels = encoded.shape
            grid_size = int(math.isqrt(token_count))
            if grid_size * grid_size != token_count:
                raise ValueError(f"Cannot reshape token sequence of length {token_count} into a square grid.")
            encoded = encoded.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size)
        elif self.encoder_mode == "tokens":
            encoded = self.encoder.forward_features(normalized)
            if encoded.ndim == 3:
                if self._has_cls_token(encoded.shape[1]):
                    encoded = encoded[:, 1:, :]
                batch_size, token_count, channels = encoded.shape
                grid_size = int(math.isqrt(token_count))
                if grid_size * grid_size != token_count:
                    raise ValueError(f"Cannot reshape token sequence of length {token_count} into a square grid.")
                encoded = encoded.transpose(1, 2).reshape(batch_size, channels, grid_size, grid_size).contiguous()
        else:
            encoded = [feature.contiguous() for feature in self.encoder(normalized)]
        if isinstance(encoded, list):
            return encoded
        return encoded.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, skip_features = self.encode_features(x)
        target_height, target_width = x.shape[-2:]

        if self.decoder_mode == "skip":
            for skip_projection, refine_block, skip_features_map in zip(
                self.skip_projections,
                self.skip_refine_blocks,
                reversed(skip_features),
                strict=True,
            ):
                features = F.interpolate(features, size=skip_features_map.shape[-2:], mode="bilinear", align_corners=False)
                features = features + skip_projection(skip_features_map)
                features = refine_block(features)

        for block in self.decoder_blocks:
            next_height = min(target_height, features.shape[-2] * 2)
            next_width = min(target_width, features.shape[-1] * 2)
            features = F.interpolate(features, size=(next_height, next_width), mode="bilinear", align_corners=False)
            features = block(features)
        features = F.interpolate(features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.output_head(features)

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    @staticmethod
    def _has_cls_token(token_count: int) -> bool:
        grid_tokens = token_count - 1
        grid_size = int(math.isqrt(grid_tokens))
        return grid_size * grid_size == grid_tokens
