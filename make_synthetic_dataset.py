import argparse
import datetime
import json
import pathlib

from nsw_da_medical_image.dataset_util.enums import Phase, FocalPlane
from nsw_da_medical_image.stable_diffusion.test_diffusion import generate


def fetch_most_recent_in(path: pathlib.Path):
    "return the most recently modified file in a directory"
    most_recent: pathlib.Path | None = None

    def _last_modified(p: pathlib.Path):
        return p.stat().st_mtime_ns

    for item in path.iterdir():
        if not item.is_file():
            continue
        if most_recent is None:
            most_recent = item
            continue
        if _last_modified(item) > _last_modified(most_recent):
            most_recent = item

    if most_recent is None:
        raise ValueError(f"no file found in {path}")

    return most_recent


def get_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset using a fine-tuned Stable Diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # either pick the last one in a directory or give a specific model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path", type=str, help="path to the saved model to use"
    )
    model_group.add_argument(
        "--latest-model-in",
        type=str,
        help="use the latest saved model in this directory",
    )

    # conditions for the generation
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=[p.label for p in Phase],
        help="the phase corresponding to the prompt and/or the model",
    )
    parser.add_argument(
        "--focal-plane",
        type=str,
        default=FocalPlane.F_0.pretty,
        choices=[fp.pretty for fp in FocalPlane],
        help="the focal plane corresponding to the prompt and/or the model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a grayscale microscopic image of human embryo at phase {phase}",
        help="prompt to generate images. May contain '{phase}' and '{focal_plane}' in which case they will be replaced with the given values.",
    )

    # generation settings, is there anything else ?
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="number of images to generate",
    )

    # where the images will be stored
    parser.add_argument(
        "--synthetic-dataset-parent",
        type=str,
        default="/App/data/all-synthetic-runs",
        help="directory where all synthetic runs should be saved",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="name of the dataset",
    )

    return parser.parse_args(args)


def main():
    args = get_args()

    if not args.prompt:
        raise ValueError(f"invalid value for {args.prompt=!r}")
    prompt = str(args.prompt)

    # format {phase} and {focal_plane}
    format_kwargs = {}
    if prompt.find("{phase}") != -1:
        format_kwargs["phase"] = args.phase
    if prompt.find("{focal_plane}") != -1:
        format_kwargs["focal_plane"] = args.focal_plane
    prompt = prompt.format(**format_kwargs)

    # find the model
    if args.model_path:
        model_path = pathlib.Path(args.model_path)
    else:
        model_path = fetch_most_recent_in(pathlib.Path(args.latest_model_in))
    if not model_path.is_file():
        raise ValueError(f"model at {model_path} is not a valid file")

    # check the synthetic dataset directory
    synthetic_parent = pathlib.Path(args.synthetic_dataset_parent)
    if not synthetic_parent.is_dir():
        raise ValueError(f"{synthetic_parent=} must exist")
    synthetic_dir = synthetic_parent / str(args.dataset_name)
    if synthetic_dir.exists():
        if len(list(synthetic_dir.iterdir())) > 0:
            raise ValueError(f"{synthetic_dir=} is not empty, aborting")
    else:
        synthetic_dir.mkdir()

    # build the metadata
    metadata_dict = {
        "datetime": datetime.datetime.now().isoformat(),
        "prompt": args.prompt,
        "phase": args.phase,
        "focal-plane": args.focal_plane,
    }
    with open(synthetic_dir / "metadata.json", "w", encoding="utf8") as metadata_file:
        json.dump(metadata_dict, metadata_file)

    # generate images
    generate(
        model_path,
        prompt,
        args.num_images,
        image_destination=synthetic_dir,
    )


if __name__ == "__main__":
    main()
