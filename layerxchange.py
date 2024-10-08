import torch
from model import ResNet9


model = ResNet9(3, 10)
model.load_state_dict(torch.load('./model_weights0.pth'))
model_tuned = ResNet9(3, 10)
model_tuned.load_state_dict(torch.load('./model_weights01.pth'))
with torch.no_grad():
    model_tuned.layer3_residual2[0].weight = model.layer3_residual2[0].weight
torch.save(model_tuned.state_dict(), 'model_weights_exchange.pth')
