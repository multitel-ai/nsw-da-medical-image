import os
import argparse
from nsw_da_medical_image.classifier.train import run_train

def main(args):
    assert os.path.exists(args.data_dir), f"Path not found: {args.data_dir}"
    assert args.architecture in ('densenet121', 'resnet50'), f"Invalid 'architecture' value: {args.architecture}" 
    assert os.path.exists(args.json_file), f"Json file not found: {args.json_file}"
    assert os.path.exists(args.save_dir), f"Path not found: {args.json_file}"
    
    run_train(architecture=args.architecture,
              num_epochs=args.num_epochs,
              lr=args.lr, 
              batch_size=args.batch_size,
              weights=args.pretrained_weights, 
              data_dir=args.data_dir,
              json_file=args.json_file,
              wandb_project_name=args.wandb_project, 
              wandb_run_name=args.name,
              freeze=args.freeze)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the training of a classifier model for the embryo dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
            
    parser.add_argument('--architecture', type=str, default='resnet50',
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Max number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--data_dir', type=str, default='/App/data/extracted',
                        help='Data path')
    parser.add_argument('--json_file', type=str, default="split.json",
                        help='Json file with train/val split')
    parser.add_argument('--pretrained_weights', type=str, default="pretrained",
                        help='Set pretrained weights')
    parser.add_argument('--wandb_project', type=str, default='classifier', 
                        help='wandb project name')
    parser.add_argument('--name', type=str, default='Give me a name !', 
                        help='wandb run name')
    parser.add_argument('--save_dir', type=str, default='/App/models', 
                        help='Directory to save the models to')
    parser.add_argument('--freeze', action='store_true', default=False, 
                        help='Whether to only train the last layer or not.')
    
    
    args = parser.parse_args()
    
    main(args)
