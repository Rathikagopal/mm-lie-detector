import torch
import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.layer_1 = nn.Sequential(nn.Linear(478 * 3, 1024, bias=True), nn.Dropout(0.2), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(1024, 512, bias=True), nn.Dropout(0.2), nn.ReLU())
        self.layer_3 = nn.Sequential(nn.Linear(512, 256, bias=True), nn.Dropout(0.2), nn.ReLU())
        self.layer_4 = nn.Sequential(nn.Linear(256, 128, bias=True), nn.Dropout(0.1), nn.ReLU())
        self.layer_5 = nn.Sequential(nn.Linear(128, 64, bias=True), nn.Dropout(0.1), nn.ReLU())
        self.fc = nn.Linear(64, 3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.fc(x)
        return x


def regressor(device, pretrained=True):
    model = Regressor().to(device)

    if pretrained:
        state = torch.load("weights/regressor.pth", map_location=torch.device(device))
        model.load_state_dict(state)

    return model
