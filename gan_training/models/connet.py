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

        # Submodules
        # nlayers = int(np.log2(size / s0))
        # self.nf0 = min(nf_max, nf * 2**nlayers)
        #
        # self.embedding = nn.Embedding(nlabels, embed_size)
        # self.fc = nn.Linear(z_dim + embed_size, self.nf0*s0*s0)
        #
        # blocks = []
        # for i in range(nlayers):
        #     nf0 = min(nf * 2**(nlayers-i), nf_max)
        #     nf1 = min(nf * 2**(nlayers-i-1), nf_max)
        #     blocks += [
        #         ResnetBlock(nf0, nf1),
        #         nn.Upsample(scale_factor=2)
        #     ]
        #
        # blocks += [
        #     ResnetBlock(nf, nf),
        # ]
        #
        # self.resnet = nn.Sequential(*blocks)
        # self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

        # self.dense = torch.nn.Linear(z_dim, 512 * 4 * 4)
        # model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.BatchNorm2d(256)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.BatchNorm2d(128)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)]
        # model += [torch.nn.BatchNorm2d(64)]
        # model += [torch.nn.ReLU(True)]
        # model += [torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=True),
        #           torch.nn.Tanh()]
        self.dense = torch.nn.Linear(z_dim, 512 * 4 * 4)
        model = [torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model += [torch.nn.ReLU(True)]
        model += [torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)]
        model += [torch.nn.ReLU(True)]
        model += [torch.nn.Conv2d(128, 3, kernel_size=4, stride=2, padding=1, bias=True),
                  torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)


    def forward(self, z, y):
        assert(z.size(0) == y.size(0))
        batch_size = z.size(0)

        # if y.dtype is torch.int64:
        #     yembed = self.embedding(y)
        # else:
        #     yembed = y
        #
        # yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        #
        # yz = torch.cat([z, yembed], dim=1)



        out = self.dense(z)
        out = out.view(batch_size, 512, 4, 4)
        out = self.model(out)



        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, nlabels, size, embed_size=256, nfilter=64, nfilter_max=1024):
        super().__init__()
        self.embed_size = embed_size
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        #
        # # Submodules
        # nlayers = int(np.log2(size / s0))
        # self.nf0 = min(nf_max, nf * 2**nlayers)
        #
        # blocks = [
        #     ResnetBlock(nf, nf)
        # ]
        #
        # for i in range(nlayers):
        #     nf0 = min(nf * 2**i, nf_max)
        #     nf1 = min(nf * 2**(i+1), nf_max)
        #     blocks += [
        #         nn.AvgPool2d(3, stride=2, padding=1),
        #         ResnetBlock(nf0, nf1),
        #     ]
        #
        # self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        # self.resnet = nn.Sequential(*blocks)
        # self.fc = nn.Linear(self.nf0*s0*s0, nlabels)

        model = [spectral_norm(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True),
                 spectral_norm(torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True),

                 spectral_norm(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True),
                 spectral_norm(torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True),

                 spectral_norm(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True),
                 spectral_norm(torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True),

                 spectral_norm(torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)),
                 torch.nn.LeakyReLU(0.2, inplace=True)]
        # model = [torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
        #          torch.nn.LeakyReLU(0.2, inplace=True),
        #
        #          torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
        #          torch.nn.LeakyReLU(0.2, inplace=True),
        #
        #          torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
        #          torch.nn.LeakyReLU(0.2, inplace=True)]

        self.model = torch.nn.Sequential(*model)
        self.dense = torch.nn.Linear(512 * 4 * 4, nlabels)

    def forward(self, x, y):
        assert(x.size(0) == y.size(0))
        batch_size = x.size(0)

        out = self.model(x).view(-1, 512 * 4 * 4)
        out = self.dense(out)
        print(out.size())
        print('1')

        index = Variable(torch.LongTensor(range(out.size(0))))
        if y.is_cuda:
            index = index.cuda()
        print(y)
        out = out[index, y]

        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out
