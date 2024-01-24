import argparse
import json
import pathlib

import nsw_da_medical_image.dataset_util.image_folder as img_f
import nsw_da_medical_image.dataset_util as du

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="extracted dataset")
parser.add_argument("--split-file", type=str, required=True, help="path to split.json")
parser.add_argument(
    "--set", type=str, choices=["train", "test", "eval"], required=True, help="which set to use",
)
parser.add_argument(
    "--phases",
    type=str,
    choices=["all"] + [p.label for p in du.Phase],
    nargs="+",
    required=True,
    help="which phases should be included in the image folder",
)
parser.add_argument(
    "--image-folder-parent", type=str, required=True, help="where the image folder will be created"
)
parser.add_argument(
    "--image-folder-name",
    type=str,
    required=False,
    help="a name for the image folder (opt.) a UUIDv4 will be used if none",
)

args = parser.parse_args()

with open(args.split_file, "r", encoding="utf8") as split_file:
    split_data: dict[str, list[str]] = json.load(split_file)

try:
    videos = [du.Video(vid) for vid in split_data[args.set]]
except KeyError as e:
    raise RuntimeError(
        f"set {args.set!r} not found in split.json: {split_data.keys()}"
    ) from e
except ValueError as e:
    raise RuntimeError("at least one video is not properly recognized") from e

try:
    if args.phases == ["all"]:
        phases = list(du.Phase)
    else:
        phases = [du.Phase(p) for p in args.phases]
except ValueError as e:
    raise RuntimeError("at least one phase is not properly recognized") from e

img_f.make_image_folder_every_phase_vid(
    pathlib.Path(args.dataset),
    pathlib.Path(args.image_folder_parent),
    args.image_folder_name,
    phases,
    videos,
    [du.FocalPlane.F_0],
    on_exist="raise",
)
