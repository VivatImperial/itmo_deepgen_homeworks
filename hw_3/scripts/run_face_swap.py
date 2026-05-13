from __future__ import annotations

import argparse
import logging

import torch
from tqdm.auto import tqdm

from hw_3.scripts._common import parse_layer_range
from hw_3.src.face import ArcFaceIdentity
from hw_3.src.config import ensure_parent_dir
from hw_3.src.images import load_rgb_image
from hw_3.src.latents import load_w_plus, save_w_plus
from hw_3.src.logging_utils import configure_logging
from hw_3.src.stylegan import StyleGANGenerator, image_to_tensor, tensor_to_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize W+ for ArcFace-guided face swap.")
    parser.add_argument("--source-image", required=True, help="Image whose identity should be transferred.")
    parser.add_argument("--target-latent", required=True, help="Target W+ latent preserving pose and color.")
    parser.add_argument("--output-latent", required=True, help="Output swapped W+ latent.")
    parser.add_argument("--output-image", help="Optional rendered face swap image path.")
    parser.add_argument("--stylegan-checkpoint", required=True, help="Path to StyleGAN generator checkpoint.")
    parser.add_argument("--arcface-checkpoint", required=True, help="Path to ArcFace checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Optimization learning rate.")
    parser.add_argument("--image-size", type=int, default=112, help="ArcFace input size.")
    parser.add_argument("--identity-weight", type=float, default=1.0, help="ArcFace identity loss weight.")
    parser.add_argument("--latent-weight", type=float, default=0.1, help="Target latent preservation weight.")
    parser.add_argument(
        "--layers",
        default="4-10",
        help="Inclusive W+ layer range to optimize. Keeps other layers fixed to preserve pose/style.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    device = torch.device(args.device)
    logging.info("Loading StyleGAN checkpoint from %s on %s", args.stylegan_checkpoint, device)
    generator = StyleGANGenerator.load(args.stylegan_checkpoint, device=device)
    logging.info("Loading ArcFace checkpoint from %s", args.arcface_checkpoint)
    identity = ArcFaceIdentity.load(args.arcface_checkpoint, device=device)
    logging.info("Loading source image %s and target latent %s", args.source_image, args.target_latent)
    source_image = load_rgb_image(args.source_image)
    source_tensor = image_to_tensor(source_image, image_size=args.image_size).to(device)
    target = load_w_plus(args.target_latent, map_location=device).to(device)
    layer_range = parse_layer_range(args.layers)
    layer_mask = torch.zeros_like(target)
    layer_indices = list(layer_range.indices())
    layer_mask[:, layer_indices, :] = 1.0
    optimized = target.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized], lr=args.learning_rate)

    step_iterator = tqdm(range(args.steps), desc="face swap", disable=args.quiet)
    for _ in step_iterator:
        optimizer.zero_grad(set_to_none=True)
        generated = generator.synthesis(optimized)
        if generated.shape[-2:] != source_tensor.shape[-2:]:
            generated = torch.nn.functional.interpolate(
                generated,
                size=source_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = args.identity_weight * identity.identity_loss(generated, source_tensor)
        loss = loss + args.latent_weight * torch.nn.functional.mse_loss(optimized, target)
        loss.backward()
        if optimized.grad is not None:
            optimized.grad.mul_(layer_mask)
        optimizer.step()
        with torch.no_grad():
            optimized.copy_(target + (optimized - target) * layer_mask)
        step_iterator.set_postfix(loss=f"{loss.item():.4f}")

    save_w_plus(optimized.detach().cpu(), args.output_latent)
    logging.info("Saved swapped latent to %s", args.output_latent)
    if args.output_image:
        image = tensor_to_image(generator.synthesis(optimized.detach()))
        image.save(ensure_parent_dir(args.output_image))
        logging.info("Saved swapped image to %s", args.output_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
