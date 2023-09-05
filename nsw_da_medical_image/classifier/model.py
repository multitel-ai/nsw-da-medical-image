from torchvision.models import densenet121, densenet
from torch import nn
from torch.nn import init
from torch import load as tload
import os

def build_model(path: str = None) -> densenet.DenseNet:
    """
    Returns the model and loads the weights from path if path is not None.
    If path is None, builds the model from scratch (with randomly initialized weights)
    """
    
    if path == "pretrained":
        os.environ['TORCH_HOME'] = '/App/models'
        model = densenet121(weights="DenseNet121_Weights.IMAGENET1K_V1")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 16)
        
    elif path is not None:
        model = densenet121(num_classes=16)
        model.load_state_dict(tload(path))

    else: # initialize weights with xavier init
        model = densenet121(num_classes=16)
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    
    return model



if __name__ == '__main__':
    model = build_model(path = "pretrained")
    print(type(model))
