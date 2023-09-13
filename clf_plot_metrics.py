import os
import argparse
from nsw_da_medical_image.classifier.plot_metrics import compare_metrics

def main(args):
    for d in args.directories:
        assert os.path.exists(d), f"Path not found: {d}"
    assert os.path.exists(args.save_dir), f"Path not found: {args.save_dir}"
    
    compare_metrics(args.directories, args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Plots metrics of multiple models to compare them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--directories", type=str, nargs='*', help="Paths to directories")
    parser.add_argument("--save_dir", type=str, help="Path to directory where to save the plots")

    args = parser.parse_args()
    
    main(args)
    