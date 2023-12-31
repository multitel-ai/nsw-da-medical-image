import argparse

from nsw_da_medical_image.synth_img_metrics import compute_synth_img_metrics

ALLOWED_EXT = ["jpg","jpeg"]


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics for synthetic image data using a specified deep learning model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orig_data_path", type=str,help="Path to the original data.",default="/App/data/extracted/embryo_dataset")
    parser.add_argument("--synth_data_path", type=str,help="Path to the synthetic data. Mandatory.")    
    parser.add_argument("--orig_data_annot_folder",type=str,help="Path to the folder containing the 'XXX_phases.csv' files.",default="/App/data/extracted/embryo_dataset_annotations/")
    parser.add_argument("--result_fold_path",type=str,default="/App/results/")
    parser.add_argument("--split_file_path",type=str,help="Path to the split.json file to only run inference on the test data.",default="split.json")

    parser.add_argument("--model_weights_path", type=str,help="Path to the model. Mandatory except in debug mode, in which case imagenet weights are used.")    
    parser.add_argument("--model_architecture",type=str,default="resnet50")

    parser.add_argument("--debug",action="store_true",help="Debug mode. Only uses the first dimensions of the features and only runs a few batches.")
    parser.add_argument("--batch_size",type=int,default=50)
    parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--num_classes",type=int,default=16)
    parser.add_argument("--max_dataset_size",type=int,default=5000)

    args = parser.parse_args()

    compute_synth_img_metrics(args.model_weights_path,args.synth_data_path,args.model_architecture,args.result_fold_path,args.orig_data_path,args.orig_data_annot_folder,args.split_file_path,args.debug,args.num_classes,args.max_dataset_size,args.batch_size,args.num_workers)

if __name__ == "__main__":
    main()