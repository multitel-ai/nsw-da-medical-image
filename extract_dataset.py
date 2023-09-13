import argparse
import pathlib

from nsw_da_medical_image.dataset_util.extract_dataset import extract_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract the dataset to prepare for other scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--archives", type=str, default="/App/data/archives", help="archived dataset")
    parser.add_argument("--extracted", type=str, default="/App/data/processed", help="where the dataset will be extracted")
    args = parser.parse_args()

    extract_dataset(
        pathlib.Path(args.archives),
        pathlib.Path(args.extracted),
    )
