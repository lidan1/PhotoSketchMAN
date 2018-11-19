from __future__ import print_function

import numpy as np
from PIL import Image
import inspect, re
import os
from collections import OrderedDict
import random
import torch
from torch.autograd import Variable

import torch
from torch import nn
from torchvision import models, transforms

# from config import vgg_path
import torch.utils.model_zoo as model_zoo


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

    # image_numpy = image_tensor[0].cpu().float().numpy()
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def get_current_visuals(input, output, input_rec, output_gt):
    IN = tensor2im(input.data)
    OUT = tensor2im(output.data)
    IN_REC =  tensor2im(input_rec.data)
    OUT_GT =  tensor2im(output_gt.data)
    return OrderedDict([
                        ('IN', IN),
                        ('OUT', OUT),
                        ('IN_REC', IN_REC),
                        ('OUT_GT', OUT_GT),
                        ])



def get_current_visuals1(H, R):
    HR = tensor2im(H.data)
    LRSR =  tensor2im(R.data)

    return OrderedDict([
                        ('HR', HR),
                        ('REC', LRSR),
                        ])

def get_current_visuals2(L):
    LR = tensor2im(L.data)

    return OrderedDict([
                        ('LR', LR),
                        ])

def get_feamap_visuals(feamap,width, depth):
    Dict = []
    ncol = 8
    nrow = depth/ncol
    canvas=np.zeros((width*nrow, width*ncol))

    name = 'name'
    feamap = feamap.data[0]

    # print(feamap[0:8].cpu().float().numpy().shape)
    for i in range(0,nrow):
        l = np.concatenate(feamap[ncol*i:ncol*i + ncol].cpu().float().numpy(),1)
        canvas[i*width:(i+1)*width,:] = l

    Dict.append((name, canvas))
    return OrderedDict(Dict)

def get_feamap_visuals_vgg(feamap, width, depth):
    Dict = []
    ncol = 8
    nrow = depth / ncol
    canvas = np.zeros((width * nrow, width * ncol))

    name = 'name'
    feamap = feamap.data[0]

    # print(feamap[0:8].cpu().float().numpy().shape)
    for i in range(0, nrow):
        l = np.concatenate(feamap[ncol * i:ncol * i + ncol].cpu().float().numpy(), 1)
        canvas[i * width:(i + 1) * width, :] = l

    Dict.append((name, canvas))
    return OrderedDict(Dict)

    # save image to the disk
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path,'png')


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images


def save_network(self, network, network_label, epoch_label, gpu_ids):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(self.save_dir, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if len(gpu_ids) and torch.cuda.is_available():
        network.cuda(device_id=gpu_ids[0])


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16()
        vgg16 = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        vgg.load_state_dict(model_zoo.load_url(vgg16))
        # vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*(list(vgg.features.children())[:36])).eval()
        self.vgg = nn.DataParallel(self.vgg, device_ids=[0])

        self.mse = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        return self.mse(self.vgg(input), self.vgg(target).detach())


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LRTransformTest(object):
    def __init__(self, shrink_factor):
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.shrink_factor = shrink_factor

    def __call__(self, hr_tensor):
        hr_img = self.to_pil(hr_tensor)
        w, h = hr_img.size
        lr_scale = transforms.Scale(int(min(w, h) / self.shrink_factor), interpolation=3)
        hr_scale = transforms.Scale(min(w, h), interpolation=3)
        lr_img = lr_scale(hr_img)
        hr_restore_img = hr_scale(lr_img)
        return self.to_tensor(lr_img), self.to_tensor(hr_restore_img)


class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, x):
        return (((x[:, :, :-1, :] - x[:, :, 1:, :]) ** 2 + (x[:, :, :, :-1] - x[:, :, :, 1:]) ** 2) ** 1.25).mean()


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
