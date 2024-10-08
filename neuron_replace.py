import torch
from model import ResNet9


neuron_index_to_train = 3

model = ResNet9(3, 10)
model.load_state_dict(torch.load('./data/model_weights.pth'))
model_tuned = ResNet9(3, 10)
model_tuned.load_state_dict(torch.load('./data/model_weights_tuned.pth'))
model_tuned2 = ResNet9(3, 10)
model_tuned2.load_state_dict(torch.load('./data/model_weights_tuned.pth'))



with torch.no_grad():
    model_tuned.layer3_residual2[0].weight = model.layer3_residual2[0].weight
    model_tuned.layer3_residual2[0].weight[neuron_index_to_train] = model.layer3_residual2[0].weight[neuron_index_to_train]

torch.save(model_tuned.state_dict(), './data/model_weights_exchange_0.pth')
