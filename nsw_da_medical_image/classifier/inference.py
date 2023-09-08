import torch
import pandas as pd
from nsw_da_medical_image.classifier.model import build_model
import nsw_da_medical_image.dataset_util as du
from torch.utils.data.dataloader import DataLoader
from nsw_da_medical_image.classifier.utils import get_dataloader
import pathlib
from tqdm import tqdm
import argparse

def inference(
        data_path: str,
        out_prob_path: str,
        out_class_path: str,
        split: str = "val",
        model_path: str | None = None,
        batch_size: int = 32,
        json_file: str = "/App/code/split.json") -> None:
    """
    Runs a trained model on data and outputs two csv files: one containing the probabilities and another containing the predicted classes.

    Args:
    data_path: path to the data
    out_prob_path: path to the csv file to be created as output for probabilities
    out_class_path: path to the csv file to be created as output for classes
    model_path: path to the trained model weights
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(model_path).to(device)
    model.eval()

    prob_df = pd.DataFrame(columns=['id'] + [f'class_prob_{i}' for i in range(16)])
    class_df = pd.DataFrame(columns=['id', 'class_pred'])
    prob_preds = []
    class_preds = []

    data_path = pathlib.Path(data_path)

    dataloader = get_dataloader(data_dir=data_path,
                                mode=val,
                                batch_size=batch_size,
                                json_file=json_file)
    
    total = len(dataloader)
    for idx, (img, phase, plane, video, frame) in tqdm(enumerate(dataloader), total=total):
        ids_images = du.dataset.label_batch(plane, video, frame)
        img = img.to(device)
        
        batch_predictions = model(img)
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(batch_predictions, dim=1)
        probabilities = probabilities.detach().cpu().numpy()

        # Get predicted class
        predicted_classes = torch.argmax(batch_predictions, dim=1)
        predicted_classes = predicted_classes.detach().cpu().numpy()

        for id, prob, pred_class in zip(ids_images, probabilities, predicted_classes):
            prob_preds.append({'id': id, **{f'class_prob_{i}': p for i, p in enumerate(prob)}})
            class_preds.append({'id': id, 'class_pred': pred_class})

        torch.cuda.empty_cache()

    prob_df = pd.concat([prob_df, pd.DataFrame(prob_preds)], ignore_index=True)
    class_df = pd.concat([class_df, pd.DataFrame(class_preds)], ignore_index=True)

    prob_df.to_csv(out_prob_path, index=False)
    class_df.to_csv(out_class_path, index=False)

def main(args):
    inference(args.data_path, args.out_prob_path, args.out_class_path, args.split,
              args.model_path, args.batch_size, args.json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a model and output results to CSV files.",
					formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("data_path", type=str, help="Path to the data.")
    parser.add_argument("out_prob_path", type=str, help="Path to the CSV file to be created as output for probabilities.")
    parser.add_argument("out_class_path", type=str, help="Path to the CSV file to be created as output for classes.")
    parser.add_argument("--split", type=str, default="val", help="Which data split (test, val). Default is 'val'.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model weights. Default is None.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the dataloader. Default is 8.")
    parser.add_argument("--json_file", type=str, default="/App/code/split.json", help="Path to the JSON file that contains data splits. Default is '/App/data/split.json'.")

    args = parser.parse_args()
    main(args)
