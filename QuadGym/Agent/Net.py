import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size=1, layers=[[64, 64], [64, 64]], activation_fun=nn.LeakyReLU(), output_size=1):
        super(NN, self).__init__()
        self.actor = nn.Sequential()
        self.critic = nn.Sequential()

        for i in range(2):
            net = layers[i]
            if i == 0:
                f_o = net[0]
                self.actor.append(nn.Linear(input_size, f_o[0]))
                self.actor.append(activation_fun)

                outputs = 0

                for x in range(len(net)):
                    layer = net[x]

                    if x+1 < len(net):
                        outputs = layer[x+1]
                    else:
                        outputs = layer[x]

                    self.actor.append(nn.Linear(layer[x], outputs))
                    self.actor.append(activation_fun)

                self.actor.append(nn.Linear(outputs, output_size))
            else:
                f_o = net[1]
                self.critic.append(nn.Linear(input_size, f_o[0]))
                self.critic.append(activation_fun)
                
                outputs = 0

                for x in range(len(net)):
                    layer = net[x]
                    
                    if x+1 < len(net):
                        outputs = layer[x+1]
                    else:
                        outputs = layer[x]

                    self.critic.append(nn.Linear(layer[x], outputs))
                    self.critic.append(activation_fun)

                self.critic.append(nn.Linear(outputs, output_size))

    def forward(self, x):
        x = self.actor(x)
        return x

model = NN(input_size=784, output_size=10, layers=[[50, 50], [50]])
x = t.randn(64, 784)
print(model(x).shape)