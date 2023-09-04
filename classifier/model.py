from torchvision.models import densenet121
from torch import nn
from nn import init

def build_model(path: str = None):
    """
    Returns the model and loads the weights from path if path is not None.
    If path is None, builds the model from scratch (with randomly initialized weights)
    """
    
    if path is not None:
        model = densenet121(num_classes=16)
        model.load_state_dict(torch.load(path))
    
    elif path == "pretrained":
        model = densenet121(pretrained=True, num_classes=16)

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
    model = build_model()
    print(model)
