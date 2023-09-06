import torch
import pandas as pd
from nsw_da_medical_image.classifier.model import build_model
import nsw_da_medical_image.dataset_util as du
from torch.utils.data.dataloader import DataLoader
from nsw_da_medical_image.classifier.utils import get_dataloader
import pathlib


def inference(
        data_path: str, 
        out_path: str, 
        model_path: str | None = None, 
        json_file:str = "/App/data/split.json") -> None:
    """
    Runs a trained model on data and outputs a csv file containing the results. 
    In this csv file, there are two columns: 'id' and 'class_pred' 
    
    Args:
    data_path: path to the data
    out_path: path to the csv file to be created as an output
    model_path: path to the trained model weights
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(model_path).to(device)
    
    df = pd.DataFrame(columns=['id', 'class_pred'])
    preds = []
    
    data_path = pathlib.Path(data_path)

    video_lst = [du.Video.AL884_2, du.Video.CA390_2]
    plane_lst = [du.FocalPlane.F_0]

    dataloader = get_dataloader(data_dir=data_path,
                                 mode='test_set',
                                 batch_size=8,
                                 json_file=json_file)

    for idx, (img, phase, plane, video, frame) in enumerate(dataloader):
        breakpoint()
        print(f"{idx=} {img.shape=} {du.dataset.label_batch(plane, video, frame)}")

        ids_images = du.dataset.label_batch(plane, video, frame) 
        batch_predictions = model(batch_images)
        
        for id, pred in zip(batch_ids, batch_predictions):
            preds.append({'id':id, 'class_pred':pred})
    
    for item in data:
        df = df.append(item, ignore_index=True)
    
    df.to_csv(out_path)

if __name__ =="__main__":
    inference("/App/data/extracted", "/App/data/results")
