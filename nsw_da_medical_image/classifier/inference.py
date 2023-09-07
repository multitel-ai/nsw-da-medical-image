import torch
import pandas as pd
from nsw_da_medical_image.classifier.model import build_model
import nsw_da_medical_image.dataset_util as du
from torch.utils.data.dataloader import DataLoader
from nsw_da_medical_image.classifier.utils import get_dataloader
import pathlib
from tqdm import tqdm


def inference(
        data_path: str,
        out_prob_path: str,
        out_class_path: str,
        model_path: str | None = None,
        batch_size: int = 8,
        json_file: str = "/App/data/split.json") -> None:
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
                                mode='test_set',
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



if __name__ =="__main__":
    # example 
    inference(data_path="/App/data/extracted", 
            out_class_path="/App/data/results/inference.csv",
            out_prob_path="/App/data/results/inference_probs.csv",
            batch_size=64)
