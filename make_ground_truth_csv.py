import argparse
import pathlib

from nsw_da_medical_image.dataset_util.dataset import NSWDataset, label_single, Phase

parser = argparse.ArgumentParser(
    description="Generate the CSV file for the ground truth of every frame",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset", type=str, default="/App/data/extracted", help="Extracted dataset")
parser.add_argument("--output", type=str, default="/App/code/ground_truth.csv", help="csv ground truth path (or parent directory)")
args = parser.parse_args()

base_path = pathlib.Path(args.dataset)
output_path = pathlib.Path(args.output)
if output_path.is_dir():
    output_path = output_path / "ground-truth.csv"
if output_path.exists():
    raise RuntimeError(f"cannot override {output_path}")


def noop(image):
    "a transform that doesn't load the PIL image to avoid wasting time"


dataset = NSWDataset(base_path, transform=noop)  # type:ignore
length = len(dataset)

with open(output_path, "w", encoding="utf8") as output_csv:
    print("identifier,phase-label,phase-index", file=output_csv)
    idx = 0
    for data_item in dataset:
        label = label_single(data_item)
        phase_idx = data_item.phase
        phase_lbl = Phase.from_idx(phase_idx).label
        print(f"{label},{phase_lbl},{phase_idx}", file=output_csv)
        idx += 1
        if idx % (length // 100) == 0:
            print(f"{idx}/{length} = {idx/length}")
