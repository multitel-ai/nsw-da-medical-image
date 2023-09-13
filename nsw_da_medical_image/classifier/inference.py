import pathlib
from tqdm import tqdm
import pandas as pd
import torch
import typing
from torch.utils.data.dataloader import DataLoader
from .model import build_model
from .. import dataset_util as du
from .utils import get_dataloader


def inference(
        data_path: str,
        out_path: str,
        json_file: str,
        split: str = typing.Literal["train","val","test"],
        architecture: str = typing.Literal['resnet50', 'densenet121'],
        model_path: str | None = None,
        batch_size: int = 32,
    ) -> None:
    """
    Runs a trained model on data and outputs two csv files: one containing the probabilities and another containing the predicted classes.

    Args:
    data_path: path to the data
    out_path: path to the output directory
    json_file: Path to the JSON file that contains data splits
    split: Data split, either 'test' or 'val'
    model_path: path to the trained model weights. If None, initializes weights.
    batch_size: Batch size for the dataloader
    """
    data_path = pathlib.Path(data_path)
    out_path = pathlib.Path(out_path)
    model_path = pathlib.Path(model_path)
    model_name = model_path.stem
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(net=architecture,path=model_path).to(device)
    model.eval()

    prob_df = pd.DataFrame(columns=['id'] + [f'class_prob_{i}' for i in range(16)])
    class_df = pd.DataFrame(columns=['id', 'class_pred'])
    prob_preds = []
    class_preds = []

    out_path.mkdir(exist_ok=True)

    dataloader = get_dataloader(data_dir=data_path,
                                mode=split,
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
        del img, batch_predictions

    prob_df = pd.concat([prob_df, pd.DataFrame(prob_preds)], ignore_index=True)
    class_df = pd.concat([class_df, pd.DataFrame(class_preds)], ignore_index=True)

    prob_df.to_csv(out_path / (model_name + "_pred-prob.csv"), index=False)
    class_df.to_csv(out_path / (model_name + "_pred-class.csv"), index=False)


