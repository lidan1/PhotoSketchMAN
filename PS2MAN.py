## original ps2man

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools

import torch.nn as nn
from torchvision import models, transforms
import sys
import argparse
from visualizer import Visualizer
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from data.data_loader import CreateDataLoader
import time
import PIL.Image
from net import networks as nets
import util
from util import AvgMeter, ImagePool


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='',
                         help='path to images (should have subfolders trainA, trainB, valA, valB, testA, testB)')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--serial_batches', action='store_true',
                         help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--train_display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--val_display_id', type=int, default=10, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--display_single_pane_ncols', type=int, default=4,
                         help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                         help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                         help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width]')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--no_flip', action='store_true', help='use dropout for the generator')
parser.add_argument('--resume', default = '')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=300,
                         help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--print_iter', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--display_iter', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--save_iter', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--ckpt_path', default = '')


opt = parser.parse_args()
print(opt)

train_visual = Visualizer(opt.train_display_id,'train',5)
val_visual = Visualizer(opt.val_display_id,'val',5)

if not os.path.exists(opt.ckpt_path):
    os.makedirs(opt.ckpt_path)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

opt.pahse = 'val'
val_data_loader = CreateDataLoader(opt)
val_dataset = val_data_loader.load_data()

# val_loader = DataLoader(val_set, batch_size=1, num_workers=12, pin_memory=True)

## define models
# 1---256x256 stage
# 2---128x128 stage
# 3---64x64 stage
GA = nets.define_G(input_nc=3,output_nc=3,ngf=64,
                    which_model_netG='resnet_9blocks', gpu_ids=[0],init_type='normal')
GB = nets.define_G(input_nc=3,output_nc=3,ngf=64,
                    which_model_netG='resnet_9blocks', gpu_ids=[0], init_type='normal')
DA1 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DA2 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DA3 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DB1 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DB2 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)
DB3 = nets.define_D(input_nc=2* 3, ndf=64,
                    which_model_netD='n_layers',gpu_ids=[0],init_type='normal', n_layers_D=1)

## resume training
idx = 0
if opt.resume:
    print('resume')
    gapath = os.path.join(opt.ckpt_path, opt.resume + '_ga.pth')
    gbpath = os.path.join(opt.ckpt_path, opt.resume + '_gb.pth')
    da1path = os.path.join(opt.ckpt_path, opt.resume + '_da1.pth')
    da2path = os.path.join(opt.ckpt_path, opt.resume + '_da2.pth')
    da3path = os.path.join(opt.ckpt_path, opt.resume + '_da3.pth')
    db1path = os.path.join(opt.ckpt_path, opt.resume + '_db1.pth')
    db2path = os.path.join(opt.ckpt_path, opt.resume + '_db2.pth')
    db3path = os.path.join(opt.ckpt_path, opt.resume + '_db3.pth')

    idx = split = opt.resume.split('_')[1]
    if not any([os.path.isfile(gapath), os.path.isfile(gbpath),
                os.path.isfile(da1path), os.path.isfile(db1path),
                os.path.isfile(da2path), os.path.isfile(db2path),
                os.path.isfile(da3path), os.path.isfile(db3path)]):
        print("=> missing checkpoint files at '{}'".format(opt.resume))

    else:
        print("=> loading checkpoint '{}'".format(opt.resume))

        # g.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.g_snapshot)))
        GA.load_state_dict(torch.load(gapath))
        GB.load_state_dict(torch.load(gbpath))
        DA1.load_state_dict(torch.load(da1path))
        DA2.load_state_dict(torch.load(da2path))
        DA3.load_state_dict(torch.load(da3path))
        DB1.load_state_dict(torch.load(db1path))
        DB2.load_state_dict(torch.load(db2path))
        DB3.load_state_dict(torch.load(db3path))

        opt.ckpt_path = os.path.join(opt.ckpt_path, 'resume_'+ idx)
        if not os.path.exists(opt.ckpt_path):
            os.makedirs(opt.ckpt_path)

GA = GA.cuda()
GB = GB.cuda()
DA1 = DA1.cuda()
DA2 = DA2.cuda()
DA3 = DA3.cuda()
DB1 = DB1.cuda()
DB2 = DB2.cuda()
DB3 = DB3.cuda()

## optimizer
optimizer_G = torch.optim.Adam(itertools.chain(GA.parameters(), GB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_A1 = torch.optim.Adam(DA1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B1 = torch.optim.Adam(DB1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizer_D_A2 = torch.optim.Adam(DA2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B2 = torch.optim.Adam(DB2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizer_D_A3 = torch.optim.Adam(DA3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D_B3 = torch.optim.Adam(DB3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

fake_A_pool = ImagePool(50)
fake_B_pool = ImagePool(50)
fake_A64_pool = ImagePool(50)
fake_B64_pool = ImagePool(50)
fake_A128_pool = ImagePool(50)
fake_B128_pool = ImagePool(50)

Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor

## define losses
criterionGAN = nets.GANLoss(use_lsgan=not opt.no_lsgan, tensor=Tensor)
criterionCycle = torch.nn.L1Loss()
criterionIdt = torch.nn.L1Loss()
criterionRec = torch.nn.L1Loss()
criterionPatch = nets.patchloss()

scale128_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((128,128),interpolation=PIL.Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
scale64_transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((64,64),interpolation=PIL.Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

seg_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])

def update_d(netD,real,fake):
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    pred_real = netD(real.detach())
    loss_D_real = criterionGAN(pred_real, True)
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    return loss_D_real, loss_D_fake, loss_D


def train():

    da1_real_loss_record = AvgMeter()
    da1_fake_loss_record = AvgMeter()
    da1_loss_record = AvgMeter()
    da2_real_loss_record = AvgMeter()
    da2_fake_loss_record = AvgMeter()
    da2_loss_record = AvgMeter()
    da3_real_loss_record = AvgMeter()
    da3_fake_loss_record = AvgMeter()
    da3_loss_record = AvgMeter()

    db1_real_loss_record = AvgMeter()
    db1_fake_loss_record = AvgMeter()
    db1_loss_record = AvgMeter()
    db2_real_loss_record = AvgMeter()
    db2_fake_loss_record = AvgMeter()
    db2_loss_record = AvgMeter()
    db3_real_loss_record = AvgMeter()
    db3_fake_loss_record = AvgMeter()
    db3_loss_record = AvgMeter()

    ga_ad1_loss_record = AvgMeter()
    ga_ad2_loss_record = AvgMeter()
    ga_ad3_loss_record = AvgMeter()
    gb_ad1_loss_record = AvgMeter()
    gb_ad2_loss_record = AvgMeter()
    gb_ad3_loss_record = AvgMeter()

    ga_loss_record = AvgMeter()
    gb_loss_record = AvgMeter()

    cyc_a1_loss_record = AvgMeter()
    cyc_a2_loss_record = AvgMeter()
    cyc_a3_loss_record = AvgMeter()
    cyc_b1_loss_record = AvgMeter()
    cyc_b2_loss_record = AvgMeter()
    cyc_b3_loss_record = AvgMeter()

    syn_a1_loss_record = AvgMeter()
    syn_a2_loss_record = AvgMeter()
    syn_a3_loss_record = AvgMeter()
    syn_b1_loss_record = AvgMeter()
    syn_b2_loss_record = AvgMeter()
    syn_b3_loss_record = AvgMeter()

    g_loss_record = AvgMeter()

    total_steps = 0
    for epoch in range(1, opt.niter + opt.niter_decay +1):
        for i, data in enumerate(dataset):

            input_A = data['A'].float()
            input_B = data['B'].float()

            inputAimg = input_A
            inputBimg = input_B

            input_Aimg = util.tensor2im(inputAimg)
            input_Bimg = util.tensor2im(inputBimg)
            input_A128 = torch.unsqueeze(scale128_transform(input_Aimg),0)

            input_A64 = torch.unsqueeze(scale64_transform(input_Aimg),0)
            input_B128 = torch.unsqueeze(scale128_transform(input_Bimg),0)
            input_B64 = torch.unsqueeze(scale64_transform(input_Bimg),0)

            real_A = Variable(inputAimg).cuda()
            real_A128 = Variable(input_A128).cuda()
            real_A64 = Variable(input_A64).cuda()

            real_B = Variable(inputBimg).cuda()
            real_B128 = Variable(input_B128).cuda()
            real_B64 = Variable(input_B64).cuda()

            ### PHOTO-->SKETCH-->PHOTO
            fake_B64, fake_B128, fake_B = GA(real_A)
            rec_A64, rec_A128, rec_A = GB(fake_B)
            ### SKETCH-->PHOTO-->SKETCH
            fake_A64, fake_A128, fake_A = GB(real_B)
            rec_B64, rec_B128, rec_B = GA(fake_A)

### update D first
            DA1.zero_grad()
            DA2.zero_grad()
            DA3.zero_grad()
            DB1.zero_grad()
            DB2.zero_grad()
            DB3.zero_grad()

            fakeA = fake_A_pool.query(torch.cat((real_B, fake_A), 1).data)
            realA = torch.cat((real_B, real_A),1)
            loss_DA256_real,loss_DA256_fake,loss_DA256 = update_d(DA1,realA,fakeA)
            optimizer_D_A1.step()
            #
            # print(real_A128)
            # print(fake_A128)
            fakeA128 = fake_A128_pool.query(torch.cat((real_B128, fake_A128), 1).data)
            realA128 = torch.cat((real_B128, real_A128),1)
            loss_DA128_real,loss_DA128_fake,loss_DA128 = update_d(DA2,realA128,fakeA128)
            optimizer_D_A2.step()

            fakeA64 = fake_A64_pool.query(torch.cat((real_B64, fake_A64), 1).data)
            realA64 = torch.cat((real_B64, real_A64),1)
            loss_DA64_real,loss_DA64_fake,loss_DA64 = update_d(DA3,realA64,fakeA64)
            optimizer_D_A3.step()

            fakeB = fake_B_pool.query(torch.cat((real_A, fake_B), 1).data)
            realB = torch.cat((real_A, real_B),1)
            loss_DB256_real,loss_DB256_fake,loss_DB256 = update_d(DB1,realB,fakeB)
            optimizer_D_B1.step()

            fakeB128 = fake_B128_pool.query(torch.cat((real_A128, fake_B128), 1).data)
            realB128 = torch.cat((real_A128, real_B128),1)
            loss_DB128_real,loss_DB128_fake,loss_DB128 = update_d(DB2,realB128,fakeB128)
            optimizer_D_B2.step()

            fakeB64 = fake_B64_pool.query(torch.cat((real_B64, fake_B64), 1).data)
            realB64 = torch.cat((real_A64, real_B64),1)
            loss_DB64_real,loss_DB64_fake,loss_DB64 = update_d(DB3,realB64,fakeB64)
            optimizer_D_B3.step()


## update G
            GA.zero_grad()
            GB.zero_grad()

            # First, G(A) should fake the discriminator
            pred_fakeB = DA1(fakeB)
            loss_GA_GAN = criterionGAN(pred_fakeB, True)
            pred_fakeB128 = DA2(fakeB128)
            loss_GA_GAN128 = criterionGAN(pred_fakeB128, True)
            pred_fakeB64 = DA3(fakeB64)
            loss_GA_GAN64 = criterionGAN(pred_fakeB64, True)

            pred_fakeA = DB1(fakeA)
            loss_GB_GAN = criterionGAN(pred_fakeA, True)
            pred_fakeA128 = DB2(fakeA128)
            loss_GB_GAN128 = criterionGAN(pred_fakeA128, True)
            pred_fakeA64 = DB3(fakeA64)
            loss_GB_GAN64 = criterionGAN(pred_fakeA64, True)

            # Second, G(A) = B
            syn_A256 = criterionRec(fake_A,real_A)
            syn_A128 = criterionRec(fake_A128,real_A128)
            syn_A64  = criterionRec(fake_A64,real_A64)

            syn_B256 = criterionRec(fake_B,real_B)
            syn_B128 = criterionRec(fake_B128,real_B128)
            syn_B64  = criterionRec(fake_B64,real_B64)

            cyc_A256 = criterionRec(rec_A,real_A)
            cyc_A128 = criterionRec(rec_A128,real_A128)
            cyc_A64  = criterionRec(rec_A64,real_A64)

            cyc_B256 = criterionRec(rec_B,real_B)
            cyc_B128 = criterionRec(rec_B128,real_B128)
            cyc_B64  = criterionRec(rec_B64,real_B64)

            eta = 1
            mu = 0.7
            Lambda = 10

            loss_G =  eta * loss_GA_GAN \
                    + eta * loss_GA_GAN128\
                    + eta * loss_GA_GAN64\
                    + eta * loss_GB_GAN\
                    + eta * loss_GB_GAN128\
                    + eta * loss_GB_GAN64\
                    + mu * cyc_A64 \
                    + mu * cyc_A128 \
                    + mu * cyc_A256 \
                    + mu * cyc_B64 \
                    + mu * cyc_B128 \
                    + mu * cyc_B256 \
                    + Lambda * syn_A64 \
                    + Lambda * syn_A128 \
                    + Lambda * syn_A256 \
                    + Lambda * syn_B64 \
                    + Lambda * syn_B128 \
                    + Lambda * syn_B256 \

            loss_G.backward()
            optimizer_G.step()


            da1_loss_record.update(loss_DA256.data[0])
            da1_real_loss_record.update(loss_DA256_real.data[0])
            da1_fake_loss_record.update(loss_DA256_fake.data[0])
            da2_loss_record.update(loss_DA128.data[0])
            da2_real_loss_record.update(loss_DA128_real.data[0])
            da2_fake_loss_record.update(loss_DA128_fake.data[0])
            da3_loss_record.update(loss_DA64.data[0])
            da3_real_loss_record.update(loss_DA64_real.data[0])
            da3_fake_loss_record.update(loss_DA64_fake.data[0])

            db1_loss_record.update(loss_DB256.data[0])
            db1_real_loss_record.update(loss_DB256_real.data[0])
            db1_fake_loss_record.update(loss_DB256_fake.data[0])
            db2_loss_record.update(loss_DB128.data[0])
            db2_real_loss_record.update(loss_DB128_real.data[0])
            db2_fake_loss_record.update(loss_DB128_fake.data[0])
            db3_loss_record.update(loss_DB64.data[0])
            db3_real_loss_record.update(loss_DB64_real.data[0])
            db3_fake_loss_record.update(loss_DB64_fake.data[0])

            ga_ad1_loss_record.update(loss_GA_GAN.data[0])
            ga_ad2_loss_record.update(loss_GA_GAN128.data[0])
            ga_ad3_loss_record.update(loss_GA_GAN64.data[0])
            gb_ad1_loss_record.update(loss_GB_GAN.data[0])
            gb_ad2_loss_record.update(loss_GB_GAN128.data[0])
            gb_ad3_loss_record.update(loss_GB_GAN64.data[0])

            cyc_a1_loss_record.update(cyc_A256.data[0])
            cyc_a2_loss_record.update(cyc_A128.data[0])
            cyc_a3_loss_record.update(cyc_A64.data[0])
            cyc_b1_loss_record.update(cyc_B256.data[0])
            cyc_b2_loss_record.update(cyc_B128.data[0])
            cyc_b3_loss_record.update(cyc_B64.data[0])

            syn_a1_loss_record.update(syn_A256.data[0])
            syn_a2_loss_record.update(syn_A128.data[0])
            syn_a3_loss_record.update(syn_A64.data[0])
            syn_b1_loss_record.update(syn_B256.data[0])
            syn_b2_loss_record.update(syn_B128.data[0])
            syn_b3_loss_record.update(syn_B64.data[0])
            g_loss_record.update(loss_G.data[0])
            # print(loss_G.data[0])

            if i % opt.print_iter == 0:
                print(
                '[train]: [epoch %d], [iter %d / %d],'
                '[da1_ad_loss %.5f],[da1_real_loss %.5f],[da1_fake_loss %.5f],'
                '[da2_ad_loss %.5f],[da2_real_loss %.5f],[da2_fake_loss %.5f],'
                '[da3_ad_loss %.5f],[da3_real_loss %.5f],[da3_fake_loss %.5f],'
                '[db1_ad_loss %.5f],[db1_real_loss %.5f],[db1_fake_loss %.5f],'
                '[db2_ad_loss %.5f],[db2_real_loss %.5f],[db2_fake_loss %.5f],'
                '[db3_ad_loss %.5f],[db3_real_loss %.5f],[db3_fake_loss %.5f],'
                '[ga_ad1_loss %.5f],[ga_ad2_loss %.5f],[ga_ad3_loss %.5f],'
                '[gb_ad1_loss %.5f],[gb_ad2_loss %.5f],[gb_ad3_loss %.5f],'
                '[syn_a1_loss %.5f],[syn_a2_loss %.5f],[syn_a3_loss %.5f],'
                '[syn_b1_loss %.5f],[syn_b2_loss %.5f],[syn_b3_loss %.5f],'
                '[cyc_a1_loss %.5f],[cyc_a2_loss %.5f],[cyc_a3_loss %.5f],'
                '[cyc_b1_loss %.5f],[cyc_b2_loss %.5f],[cyc_b3_loss %.5f],'
                '[g_loss %.5f]' % \
                (epoch + 1, i + 1, dataset_size,
                 da1_loss_record.avg, da1_real_loss_record.avg, da1_fake_loss_record.avg,
                 da2_loss_record.avg, da2_real_loss_record.avg, da2_fake_loss_record.avg,
                 da3_loss_record.avg, da3_real_loss_record.avg, da3_fake_loss_record.avg,
                 db1_loss_record.avg, db1_real_loss_record.avg, db1_fake_loss_record.avg,
                 db2_loss_record.avg, db2_real_loss_record.avg, db2_fake_loss_record.avg,
                 db3_loss_record.avg, db3_real_loss_record.avg, db3_fake_loss_record.avg,
                 ga_ad1_loss_record.avg, ga_ad2_loss_record.avg, ga_ad3_loss_record.avg,
                 gb_ad1_loss_record.avg, gb_ad2_loss_record.avg, gb_ad3_loss_record.avg,
                 syn_a1_loss_record.avg, syn_a2_loss_record.avg, syn_a3_loss_record.avg,
                 syn_b1_loss_record.avg, syn_b2_loss_record.avg, syn_b3_loss_record.avg,
                 cyc_a1_loss_record.avg, cyc_a2_loss_record.avg, cyc_a3_loss_record.avg,
                 cyc_b1_loss_record.avg, cyc_b2_loss_record.avg, cyc_b3_loss_record.avg,
                 g_loss_record.avg)
                )

            iter = int(idx) + epoch * dataset_size + i

            if i % opt.display_iter == 0:
                A256 = util.get_current_visuals(real_A,fake_B,rec_A,real_B)
                A128 = util.get_current_visuals(real_A128, fake_B128, rec_A128, real_B128)
                A64 = util.get_current_visuals(real_A64, fake_B64, rec_A64, real_B64)
                B256 = util.get_current_visuals(real_B, fake_A, rec_B, real_A)
                B128 = util.get_current_visuals(real_B128, fake_A128, rec_B128, real_A128)
                B64 = util.get_current_visuals(real_B64, fake_A64, rec_B64, real_A64)

                train_visual.display_current_results(A256,iter, winid=1)
                train_visual.display_current_results(A128, iter, winid=2)
                train_visual.display_current_results(A64, iter, winid=3)
                train_visual.display_current_results(B256,iter, winid=4)
                train_visual.display_current_results(B128, iter, winid=5)
                train_visual.display_current_results(B64, iter,winid=6)

                err1 = OrderedDict([
                        ('cyc_a1_loss', cyc_A256.data[0]),
                        ('cyc_b1_loss', cyc_B256.data[0]),
                        ('syn_a1_loss', syn_A256.data[0]),
                        ('syn_b1_loss', syn_B256.data[0]),
                        ('ga_ad1_loss', loss_GA_GAN.data[0]),
                        ('gb_ad1_loss', loss_GB_GAN.data[0]),
                        ('da1_loss', loss_DA256.data[0]),
                        ('da1_real_loss', loss_DA256_real.data[0]),
                        ('da1_fake_loss', loss_DA256_fake.data[0]),
                        ('db1_loss', loss_DB256.data[0]),
                        ('db1_real_loss', loss_DB256_real.data[0]),
                        ('db1_fake_loss', loss_DB256_fake.data[0]),

                ])
                # print iter
                # print err1
                train_visual.plot_current_errors(iter, err1, winid=10)
                test()

            if i % opt.save_iter == 0 :
                snapshot_name = str(iter)
                torch.save(GA.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_ga.pth'))
                torch.save(GB.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_gb.pth'))
                torch.save(DA1.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_da1.pth'))
                torch.save(DA2.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_da2.pth'))
                torch.save(DA3.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_da3.pth'))
                torch.save(DB1.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_db1.pth'))
                torch.save(DB2.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_db2.pth'))
                torch.save(DB3.state_dict(), os.path.join(opt.ckpt_path, snapshot_name + '_db3.pth'))

def test():
    for i, data in enumerate(val_dataset):
        input_A = data['A'].float()
        input_B = data['B'].float()

        # print Apath

        inputAimg = input_A
        inputBimg = input_B

        real_A = Variable(inputAimg).cuda()

        real_B = Variable(inputBimg).cuda()

        ### PHOTO-->SKETCH-->PHOTO
        fake_B64, fake_B128, fake_B = GA(real_A)
        rec_A64, rec_A128, rec_A = GB(fake_B)
        ### SKETCH-->PHOTO-->SKETCH
        fake_A64, fake_A128, fake_A = GB(real_B)
        rec_B64, rec_B128, rec_B = GA(fake_A)

        A256 = util.get_current_visuals(real_A, fake_B, rec_A, real_B)
        B256 = util.get_current_visuals(real_B, fake_A, rec_B, real_A)
        val_visual.display_current_results(A256, iter, winid=10)
        val_visual.display_current_results(B256, iter, winid=15)
        break

if __name__ == '__main__':
    train()
