import argparse
import json
import pathlib

from nsw_da_medical_image.dataset_util import split

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, help="extracted dataset")
parser.add_argument("output", type=str, help="path for split.json")
parser.add_argument("weights", type=float, nargs="+", help="weights")
parser.add_argument("--seed", type=int, required=False)
args = parser.parse_args()

kwargs = {}
if args.seed is not None:
    kwargs["seed"] = args.seed

base_path = pathlib.Path(args.dataset)
output_path = pathlib.Path(args.output)
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
        keys.insert(1, "eval")
    case _:
        raise ValueError(f"expected 2 or 3 elements for {weights=!r}")

phase_counter = split.count_phases_in_videos(base_path)
per_phase_vid_lst = split.count_vid_per_phase(phase_counter)
sets = split.fair_random_split(per_phase_vid_lst, args.weights, **kwargs)

json_data = {k: [v.directory for v in s] for k, s in zip(keys, sets)}
json_str = json.dumps(json_data)

with open(output_path, "w", encoding="utf8") as json_file:
    json_file.write(json_str)
