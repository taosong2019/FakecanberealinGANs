import torch
from gan_training.metrics.inception_score import inception_score
from gan_training.metrics import fid_score
from gan_training import utils
import numpy as np
import os
from os import path
import torchvision

class Evaluator(object):
    def __init__(self, generator, zdist, ydist, batch_size=64,
                 inception_nsamples=60000, fid_nsamples=10000, device=None):
        # self.x_real_fid = x_real_fid
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.fid_nsamples = fid_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        # must be this code for updating evaluate.generator (by Taosong)
        self.generator.eval()
        imgs = []
        while(len(imgs) < self.inception_nsamples):
            ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )
        return score, score_std


        # num_batch = self.fid_nsamples/self.batch_size
        # fake_list = []
        # # real_list = []
        # while (len(fake_list) < num_batch):
        #     ztest = self.zdist.sample((self.batch_size,))
        #     ytest = self.ydist.sample((self.batch_size,))
        #
        #     samples = self.generator(ztest, ytest)
        #
        #
        #     fake_list.append((samples.cpu().numpy() + 1.0) / 2.0)
        #     # real_list.append((self.x_real_fid.cpu().numpy() + 1.0) / 2.0)
        #
        # fake_images = np.concatenate(fake_list)
        # fake_images = fake_images[:self.fid_nsamples]
        # # real_images = np.concatenate(real_list)
        # # real_images = real_images[:self.fid_nsamples]
        #
        # # print(fake_images)
        # # print(fake_images.size())
        #
        # # real_images = (self.x_real_fid.cpu().numpy() + 1.0) / 2.0
        # # print(real_images)
        # # print(real_images.size())
        # # Here x_real_fid, imgs are both normalized, which should be transformed into 0~1.
        #
        # fake_images = (self.x_real_fid.cpu().numpy() + 1.0) / 2.0
        # fid = inception_fid(
        #      fake_images, batch_size=50, device=self.device)





    def compute_fid_score(self, paths):
        fid= fid_score.calculate_fid_given_paths(paths=paths,device=self.device)
        return fid


    def fake_samples_store(self,it, path_fake):
        path_store = path.join(path_fake, '%08d'%it)
        if not path.exists(path_store):
            os.makedirs(path_store)

        iter_sample = 0
        num_batch = self.fid_nsamples / self.batch_size
        fake_images = []
        while (iter_sample < num_batch):
            ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))
            samples = self.generator(ztest, ytest)/2+0.5
            samples = [s.data.cpu().numpy() for s in samples]
            fake_images.extend(samples)
            iter_sample += 1
        fake_images = fake_images[:self.fid_nsamples]
        for i in range(self.fid_nsamples):
            torchvision.utils.save_image(torch.tensor(fake_images[i]), path.join(path_store, 'fake%05d.png' % i), nrow=1)
            i += 1

        return path_store

    def create_samples(self, z, y=None):
        self.generator.eval()
        batch_size = z.size(0)
        # Parse y
        if y is None:
            y = self.ydist.sample((batch_size,))
        elif isinstance(y, int):
            y = torch.full((batch_size,), y,
                           device=self.device, dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            x = self.generator(z, y)
        return x
