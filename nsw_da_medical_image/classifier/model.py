import os
from torchvision.models import densenet121, resnet50
from torch import nn, load as tload
from torch.nn.modules.module import Module

def build_model(net: str = 'resnet50', path: str = None) -> Module:
    """
    Returns the model and loads the weights from path if path is not None.
    If path is 'pretrained', loads a version of the model that was pretrained on ImageNet1K.
    If path is None, builds the model from scratch (with randomly initialized weights)
    """
    # Define the model and its pre-trained weights
    model_classes = {
        'densenet121': (densenet121, "DenseNet121_Weights.IMAGENET1K_V1"),
        'resnet50': (resnet50, "IMAGENET1K_V2")
    }

    if net not in model_classes:
        raise ValueError(f"Value for 'net' not recognized: {net}")

    model_class, weights = model_classes[net]
    
    if path == "pretrained":
        os.environ['TORCH_HOME'] = '.'
        model = model_class(weights=weights)
    elif path:
        model = model_class(num_classes=16)
        model.load_state_dict(tload(path))
    else: 
        model = model_class(num_classes=16)
        model = _initialize_weights(model)

    model = _modify_last_layer(model, net)
    
    return model

def _initialize_weights(model: Module) -> Module:
    """Initializes weights for the model."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    return model

def _modify_last_layer(model: Module, net: str) -> Module:
    """Modifies the last layer of the model."""
    if net == 'densenet121':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 16)
    elif net == 'resnet50':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 16)
    return model

if __name__ == '__main__':
    model = build_model(path="pretrained")
    print(type(model))