import torch
from torch import nn, optim
import torch.nn.functional as F


class NetDecoder(nn.Module):
    def __init__(self):
        super(NetDecoder, self).__init__()
        self.shape = (256, 256, 3)
        self.dim = 100 # dimension of the latent vector
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 2* 4 * 4 * 4 * self.dim),    # Default is bias=True
                nn.BatchNorm1d(2 * 4 * 4 * 4 * self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, 4, stride=4),
                nn.BatchNorm2d(4 * self.dim),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 4, stride=4),
                nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(2 * self.dim, 3, 4, stride=4)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * 2 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        output = output.view(-1, 3, 256, 256)
        return F.sigmoid(output)
    
    
class NetEncoder(nn.Module):
    def __init__(self):
        super(NetEncoder, self).__init__()
        self.shape = (256, 256, 3)
        self.dim = 100
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(4 * self.dim, 8 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(8 * self.dim, 16 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*4*4*self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, 3, 256, 256)
        output = self.main(input)
        output = output.view(-1, 4*4*4*4*4*self.dim)
        output = self.linear(output)
        return output.view(-1, self.dim)
    

class NetDiscriminator(nn.Module):
    def __init__(self):
        super(NetDiscriminator, self).__init__()
        self.shape = (256, 256, 3)
        self.dim = 100
        self.N = 200

        self.lin1 = nn.Linear(self.dim, self.N)
        self.lin2 = nn.Linear(self.N, self.N)
        self.lin3 = nn.Linear(self.N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x))