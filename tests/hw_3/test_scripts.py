import importlib
from pathlib import Path


SCRIPT_MODULES = [
    "hw_3.scripts.align_images",
    "hw_3.scripts.project_images",
    "hw_3.scripts.run_style_transfer",
    "hw_3.scripts.run_expression_transfer",
    "hw_3.scripts.run_face_swap",
    "hw_3.scripts.download_image_set",
    "hw_3.scripts.run_full_pipeline",
]


def test_scripts_expose_build_parser() -> None:
    for module_name in SCRIPT_MODULES:
        module = importlib.import_module(module_name)

        assert callable(module.build_parser)


def test_align_parser_accepts_required_arguments() -> None:
    parser = importlib.import_module("hw_3.scripts.align_images").build_parser()

    args = parser.parse_args(
        ["--input-dir", "raw", "--output-dir", "aligned", "--landmarks-dir", "landmarks"]
    )

    assert args.input_dir == "raw"
    assert args.output_dir == "aligned"
    assert args.landmarks_dir == "landmarks"


def test_projection_parser_accepts_required_arguments() -> None:
    parser = importlib.import_module("hw_3.scripts.project_images").build_parser()

    args = parser.parse_args(
        [
            "--input-dir",
            "aligned",
            "--output-dir",
            "projections",
            "--stylegan-checkpoint",
            "stylegan2.pkl",
            "--include",
            "person_01",
            "--max-images",
            "1",
        ]
    )

    assert args.stylegan_checkpoint == "stylegan2.pkl"
    assert args.include == ["person_01"]
    assert args.max_images == 1


def test_transfer_parsers_accept_layer_range() -> None:
    style_parser = importlib.import_module("hw_3.scripts.run_style_transfer").build_parser()
    expression_parser = importlib.import_module("hw_3.scripts.run_expression_transfer").build_parser()

    style_args = style_parser.parse_args(
        ["--source", "style.pt", "--target", "base.pt", "--output", "out.pt", "--layers", "7-17"]
    )
    expression_args = expression_parser.parse_args(
        ["--source", "smile.pt", "--target", "base.pt", "--output", "out.pt", "--layers", "4-6"]
    )

    assert style_args.layers == "7-17"
    assert expression_args.layers == "4-6"


def test_face_swap_parser_accepts_required_arguments() -> None:
    parser = importlib.import_module("hw_3.scripts.run_face_swap").build_parser()

    args = parser.parse_args(
        [
            "--source-image",
            "identity.png",
            "--target-latent",
            "target.pt",
            "--output-latent",
            "swap.pt",
            "--stylegan-checkpoint",
            "stylegan2.pkl",
            "--arcface-checkpoint",
            "arcface.pth",
            "--layers",
            "4-10",
        ]
    )

    assert args.arcface_checkpoint == "arcface.pth"
    assert args.layers == "4-10"


def test_download_parser_accepts_retry_options() -> None:
    parser = importlib.import_module("hw_3.scripts.download_image_set").build_parser()

    args = parser.parse_args(["--manifest", "manifest.json", "--retries", "3", "--delay-seconds", "1.5"])

    assert args.retries == 3
    assert args.delay_seconds == 1.5


def test_full_pipeline_parser_accepts_key_paths_and_steps() -> None:
    parser = importlib.import_module("hw_3.scripts.run_full_pipeline").build_parser()

    args = parser.parse_args(
        [
            "--stylegan-checkpoint",
            "stylegan2.pkl",
            "--arcface-checkpoint",
            "arcface.pth",
            "--device",
            "mps",
            "--projection-steps",
            "12",
            "--face-swap-steps",
            "7",
        ]
    )

    assert args.device == "mps"
    assert args.projection_steps == 12
    assert args.face_swap_steps == 7


def test_full_pipeline_clean_artifacts_root_removes_stale_subdirs(tmp_path: Path) -> None:
    module = importlib.import_module("hw_3.scripts.run_full_pipeline")
    artifacts = tmp_path / "artifacts"
    (artifacts / "debug_old").mkdir(parents=True)
    (artifacts / "debug_old" / "image.png").write_text("stale")
    (artifacts / "tables").mkdir()
    (artifacts / "tables" / "grid.png").write_text("stale")

    module._clean_artifacts_root(artifacts)

    assert artifacts.is_dir()
    assert list(artifacts.iterdir()) == []


def test_full_pipeline_shortens_manifest_ids_for_grid_labels() -> None:
    module = importlib.import_module("hw_3.scripts.run_full_pipeline")

    assert module._display_id({"id": "person_01_barack_obama"}) == "p01"
    assert module._display_id({"id": "style_02_einstein"}) == "s02"
    assert module._display_id({"id": "expression_03_serious"}) == "e03"
