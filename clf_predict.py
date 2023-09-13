import os
import argparse
from nsw_da_medical_image.classifier.inference import inference

def main(args: argparse.Namespace) -> None:
    assert os.path.exists(args.data_path), f"Path not found: {args.data_path}"
    assert os.path.exists(args.json_file), f"Json file not found: {args.json_file}"
    assert os.path.exists(args.model_path), f"Path not found: {args.model_path}"
    assert args.architecture in ('densenet121', 'resnet50'), f"Invalid 'architecture' value: {args.architecture}"  
    assert args.split in ('train', 'val', 'test'), f"Invalid 'split' value: {args.architecture}"  

    inference(
        args.data_path,
        args.out_path,
        args.json_file,
        args.split,
        args.architecture,
        args.model_path,
        args.batch_size,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run inference on a model and output results to CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--data_path", type=str, help="Path to the data.", default="/App/data/extracted")
    parser.add_argument("--out_path", type=str, help="Path to output directory.")
    parser.add_argument("--json_file", type=str, default="/App/code/split.json", help="Path to the JSON file that contains data splits.")
    parser.add_argument("--architecture", type=str, default='resnet50', help="Model architecture. Either 'densenet121' or 'resnet50'")
    parser.add_argument("--model_path", type=str, help="Path to the trained model weights.")
    parser.add_argument("--split", type=str, default="val", help="Which data split (test, val).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the dataloader.")

    args = parser.parse_args()
    main(args)