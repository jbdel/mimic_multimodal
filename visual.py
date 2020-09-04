import torch.nn as nn
import torchvision.models as models
import torch

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



class Visual(nn.Module):
    def __init__(self, args):
        super(Visual, self).__init__()
        self.args = args
        self.net = models.resnet18(pretrained=True)
        set_parameter_requires_grad(self.net, False)
        self.in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(self.in_features, 300)

    def forward(self, img):
        return self.net(img)


class Finetune(Visual):
    def __init__(self, args):
        super(Finetune, self).__init__(args)
        if args.checkpoint is not None:
            state_dict = torch.load(args.checkpoint)['state_dict']
            self.net.load_state_dict(state_dict)
        self.net.fc = torch.nn.Linear(self.in_features, 14)

    def forward(self, img):
        return self.net(img)


