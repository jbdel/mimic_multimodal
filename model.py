import torch.nn as nn
import torchvision.models as models
import torch

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Model(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Model, self).__init__()
        net = models.densenet161(pretrained=True)
        set_parameter_requires_grad(net, False)
        num_ftrs = net.classifier.in_features
        net.classifier = torch.nn.Linear(num_ftrs, 300)
        self.net = net

    def forward(self, img):
        return self.net(img)
