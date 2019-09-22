import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm


class Generator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.z_dim = z_dim


        self.dense = torch.nn.Linear(z_dim, 256 * 4 * 4)
        # model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
        #           torch.nn.Tanh()]
        model = [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model += [torch.nn.ReLU(True)]
        model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
        model += [torch.nn.ReLU(True)]
        model += [torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True),
                  torch.nn.Tanh()]
        # model = [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=True),
        #           torch.nn.Tanh()]


        self.model = torch.nn.Sequential(*model)

    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)


        out = self.dense(z)
        out = out.view(batch_size, 256, 4, 4)
        out = self.model(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max


        model = [torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
                 torch.nn.LeakyReLU(0.2, inplace=True),

                 torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
                 torch.nn.LeakyReLU(0.2, inplace=True),

                 torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
                 torch.nn.LeakyReLU(0.2, inplace=True)]

        self.model = torch.nn.Sequential(*model)
        self.dense = torch.nn.Linear(256 * 4 * 4, nlabels)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.model(x).view(-1, 256 * 4 * 4)
        out = self.dense(out)

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        out = out[index, y]

        return out


