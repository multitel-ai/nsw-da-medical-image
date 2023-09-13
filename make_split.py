import argparse
import json
import pathlib

from nsw_da_medical_image.dataset_util import split

parser = argparse.ArgumentParser(
    description="Make a fair random split between the videos",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset", type=str, default="/App/data/extracted", help="extracted dataset"
)
parser.add_argument(
    "--split_file", type=str, default="/App/code/split.json", help="path for split.json"
)
parser.add_argument(
    "--weights",
    type=float,
    nargs="+",
    default=[0.6, 0.2, 0.2],
    help="weights for the 2 or 3 splits",
)
parser.add_argument("--seed", type=int, default=1234567890, required=False)
args = parser.parse_args()

seed = args.seed

base_path = pathlib.Path(args.dataset)
output_path = pathlib.Path(args.split_file)
if output_path.is_dir():
    output_path = output_path / "split.json"
if output_path.exists():
    raise RuntimeError(f"cannot override {output_path}")


weights: list[float] = args.weights
keys = ["train", "test"]
match weights:
    case [train, test]:
        pass
    case [train, eval, test]:
        keys.insert(1, "val")
    case _:
        raise ValueError(f"expected 2 or 3 elements for {weights=!r}")

phase_counter = split.count_phases_in_videos(base_path)
per_phase_vid_lst = split.count_vid_per_phase(phase_counter)
sets = split.fair_random_split(per_phase_vid_lst, args.weights, seed)

json_data = {k: [v.directory for v in s] for k, s in zip(keys, sets)}
json_str = json.dumps(json_data)

with open(output_path, "w", encoding="utf8") as json_file:
    json_file.write(json_str)
