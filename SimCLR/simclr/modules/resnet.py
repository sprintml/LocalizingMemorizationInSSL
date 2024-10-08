import torchvision
import resnet9model

def get_resnet(name, pretrained=False):
    resnets = {
        "resnet9": resnet9model.ResNet9(3,10),
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
