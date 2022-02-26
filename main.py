import argparse
import os
import random
import time
import datetime
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


from data.dataset import ImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.patch_MLP import patchSample
from utils.utils import str2bool, init_net, set_requires_grad, sample_images
from utils.loss import PatchNCELoss, calculate_NCE_loss, GANLoss
from utils.scheduler import get_scheduler

parser = argparse.ArgumentParser()
parser.add_argument("--start-epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--epochs-decay", type=int, default=200, help="number of epochs to linearly decay learning rate to zero")
parser.add_argument('--dataroot', default='./datasets/grumpifycat', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument("--batch-size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument('--lr-policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
parser.add_argument('--lr-decay-iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("-j", "--workers", type=int, default=8, help="number of cpu threads to use during batch generation")
# for training parameters.
parser.add_argument('--input-nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output-nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
parser.add_argument('--n-layers-D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
parser.add_argument('--netF-nc', type=int, default=256)
parser.add_argument('--nce-T', type=float, default=0.07, help='temperature for NCE loss')
parser.add_argument('--num-patches', type=int, default=256, help='number of patches per layer')
parser.add_argument('--flip-equivariance',
                    type=str2bool, nargs='?', const=True, default=False,
                    help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
parser.add_argument('--init-type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
parser.add_argument('--init-gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
parser.add_argument('--no-dropout', type=str2bool, nargs='?', const=True, default=True,
                    help='no dropout for the generator')
parser.add_argument('--lambda-GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
parser.add_argument('--lambda-NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
parser.add_argument('--nce-idt', type=str2bool, nargs='?', const=True, default=True,
                    help='use NCE loss for identity mapping: NCE(G(Y), Y))')
parser.add_argument('--nce-layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
parser.add_argument('--nce-includes-all-negatives-from-minibatch',
                    type=str2bool, nargs='?', const=True, default=False,
                    help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
# data setup
parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                    help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop-size', type=int, default=256, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--random_scale_max', type=float, default=3.0,
                    help='(used for single image translation) Randomly scale the image by the specified factor as data augmentation.')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    summary = SummaryWriter()

    # STFT 인자
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # define models.
    generator = Generator(input_nc=args.input_nc, output_nc=args.output_nc, ngf=args.ngf, norm_layer=args.normG,
                          use_dropout=not args.no_dropout, no_antialias=args.no_antialias, no_antialias_up=args.no_antialias_up,
                          args=args)
    discriminator = Discriminator(input_nc=args.output_nc, ndf=args.ndf, n_layers=3, norm_layer=args.normD,
                                  no_antialias=args.no_antialias)
    patchMLP = patchSample(use_mlp=True, init_type=args.init_type, init_gain=args.init_gain,
                           nc=args.netF_nc)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
            print("rank 확인: ", args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            generator.cuda(args.gpu)
            discriminator.cuda(args.gpu)
            patchMLP.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

            generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
            discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
            patchMLP = nn.parallel.DistributedDataParallel(patchMLP, device_ids=[args.gpu])

        else:
            generator.cuda()
            discriminator.cuda()
            patchMLP.cuda()

            generator = nn.parallel.DistributedDataParallel(generator)
            discriminator = nn.parallel.DistributedDataParallel(discriminator)
            patchMLP = nn.parallel.DistributedDataParallel(patchMLP)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = generator.cuda(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        patchMLP = patchMLP.cuda(args.gpu)

    else:
        generator = nn.DataParallel(generator).cuda(args.gpu)
        discriminator = nn.DataParallel(discriminator).cuda(args.gpu)
        patchMLP = nn.DataParallel(patchMLP).cuda(args.gpu)

    # apply init_weight
    generator = init_net(generator, args.init_type, args.init_gain, initialize_weights=True)
    discriminator = init_net(discriminator, args.init_type, args.init_gain, initialize_weights=True)

    # Define Objective function.
    nce_layers = [int(i) for i in args.nce_layers.split(',')]
    criterion_GAN = GANLoss().cuda(args.gpu)
    criterionNCE = []
    for nce_layer in nce_layers:
        criterionNCE.append(PatchNCELoss(args).cuda(args.gpu))
    criterion_idt = nn.L1Loss().cuda(args.gpu)

    # Optmizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # scheduler
    scheduler_G = get_scheduler(optimizer_G, args)
    scheduler_D = get_scheduler(optimizer_D, args)

    # dataset
    dataset = ImageDataset(root=args.dataroot, unaligned=True, mode='train', args=args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        print("Sampler")
    else:
        train_sampler = None

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=(train_sampler is None),
                                             num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                             drop_last=True)

    # CUT 저자) netF(patchMLP)는 generator의 인코더 부분의 중간 feature map 추출 형태로 정의.
    # 이러한 이유 때문에, netF의 weight는 몇 개의 input image를 feedforward pass 해줘 초기화.
    optimizer_F, scheduler_F = data_dependent_initialize(dataloader=dataloader,
                                                         generator=generator,
                                                         discriminator=discriminator,
                                                         patchMLP=patchMLP,
                                                         criterion_GAN=criterion_GAN,
                                                         criterion_NCE=criterionNCE,
                                                         args=args,
                                                         nce_layers=nce_layers)

    # Print Parameters
    G_P = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    D_P = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    F_P = sum(p.numel() for p in patchMLP.parameters() if p.requires_grad)

    print("G_param", G_P)
    print("D_param", D_P)
    print("F_param", F_P)
    print("total param", G_P + D_P + F_P)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']

            generator.load_state_dict(checkpoint['G'])
            discriminator.load_state_dict(checkpoint['D'])
            patchSample.load_state_dict(checkpoint['F'])

            optimizer_G.load_state_dict(checkpoint['G_optimizer'])
            optimizer_D.load_state_dict(checkpoint['D_optimizer'])
            optimizer_F.load_state_dict(checkpoint['F_optimizer'])

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs + args.epochs_decay):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(args, dataloader, epoch, generator, discriminator, patchMLP,
              optimizer_D, optimizer_G, optimizer_F,
              scheduler_G, scheduler_D, scheduler_F,
              criterion_GAN, criterion_idt, criterionNCE,
              summary, nce_layers)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            torch.save({
                'epoch': epoch + 1,
                'G': generator.state_dict(),
                'D': discriminator.state_dict(),
                'F': patchMLP.state_dict(),
                'G_optimizer': optimizer_D.state_dict(),
                'D_optimizer': optimizer_D.state_dict(),
                'F_optimizer': optimizer_F.state_dict()
            }, "saved_models/checkpoint_%d.pth" % (epoch + 1))


def train(args, dataloader, epoch, generator, discriminator, patchMLP,
          optimizer_D, optimizer_G, optimizer_F,
          scheduler_G, scheduler_D, scheduler_F,
          criterion_GAN, criterion_idt, criterion_NCE,
          summary, nce_layers):

    generator.train()
    discriminator.train()
    patchMLP.train()

    end = time.time()

    for i, images in enumerate(dataloader):
        real_A = Variable(images['A']).cuda(args.gpu, non_blocking=True)
        real_B = Variable(images['B']).cuda(args.gpu, non_blocking=True)

        # 만약 identity mapping=True일 때 가정
        real = torch.cat((real_A, real_B), dim=0)
        if args.flip_equivariance:
            flipped_for_equivariance = (np.random.random() < 0.5)
            if flipped_for_equivariance:
                real = torch.flip(real, [3])
        else:
            flipped_for_equivariance =None

        fake = generator(real)
        fake_B = fake[:real_A.size(0)]
        idt_B = fake[real_A.size(0):]

        # Update Discrminator
        set_requires_grad(discriminator, True)
        optimizer_D.zero_grad()
        fake = fake_B.detach()
        pred_fake = discriminator(fake)
        loss_D_fake = criterion_GAN(pred_fake, False).mean()

        pred_real = discriminator(real_B)
        loss_D_real = criterion_GAN(pred_real, True)
        loss_D_real = loss_D_real.mean()

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizer_D.step()


        # update G
        set_requires_grad(discriminator, False)
        optimizer_G.zero_grad()
        optimizer_F.zero_grad()

        fake_ = fake_B
        pred_fake_ = discriminator(fake_)
        loss_G_GAN = criterion_GAN(pred_fake_, True).mean() * args.lambda_GAN

        loss_NCE = calculate_NCE_loss(args, real_A, real_B, generator, patchMLP, nce_layers, flipped_for_equivariance,
                                      criterion_NCE)
        loss_NCE_Y = calculate_NCE_loss(args, real_B, idt_B, generator, patchMLP, nce_layers, flipped_for_equivariance,
                                        criterion_NCE)
        loss_NCE_both = (loss_NCE_Y + loss_NCE) * 0.5

        loss_G = loss_G_GAN + loss_NCE_both
        loss_G.backward()

        optimizer_G.step()
        optimizer_F.step()

        niter = epoch * len(dataloader) + i
        summary.add_scalar('Train/G_loss', loss_G.item(), niter)
        summary.add_scalar('Train/D_loss', loss_D.item(), niter)
        summary.add_scalar('Train/NCE_loss', loss_NCE_both.item(), niter)
        summary.add_scalar('Train/G_GAN_loss', loss_G_GAN.item(), niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | D_loss: %f | G_loss_: %f | NCE_loss: %f | G_GAN_loss: %f"
                  % (epoch + 1, i, len(dataloader), loss_D, loss_G, loss_NCE_both,
                     loss_G_GAN))

    scheduler_G.step()
    scheduler_D.step()
    scheduler_F.step()
    lr = optimizer_G.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)
    sample_images(epoch, real_A, real_B, generator)


def data_dependent_initialize(dataloader, generator, discriminator, patchMLP, criterion_GAN, criterion_NCE,
                              args,
                              nce_layers):
    data = next(iter(dataloader))
    real_A = data['A'].cuda(args.gpu, non_blocking=True)
    real_B = data['B'].cuda(args.gpu, non_blocking=True)

    real = torch.cat((real_A, real_B), dim=0)
    if args.flip_equivariance:
        flipped_for_equivariance = (np.random.random() < 0.5)
        if flipped_for_equivariance:
            real = torch.flip(real, [3])
    else:
        flipped_for_equivariance = False

    fake = generator(real)
    fake_B = fake[:real_A.size(0)]
    idt_B = fake[real_A.size(0):]

    fake = fake_B.detach()
    pred_fake = discriminator(fake)
    loss_D_fake = criterion_GAN(pred_fake, False).mean()

    pred_real = discriminator(real_B)
    loss_D_real = criterion_GAN(pred_real, True)
    loss_D_real = loss_D_real.mean()

    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()

    fake_ = fake_B
    pred_fake_ = discriminator(fake_)
    loss_G_GAN = criterion_GAN(pred_fake_, True).mean() * args.lambda_GAN

    loss_NCE = calculate_NCE_loss(args, real_A, real_B, generator, patchMLP, nce_layers, flipped_for_equivariance,
                                  criterion_NCE)
    loss_NCE_Y = calculate_NCE_loss(args, real_B, idt_B, generator, patchMLP, nce_layers, flipped_for_equivariance,
                                  criterion_NCE)
    loss_NCE_both = (loss_NCE_Y + loss_NCE) * 0.5

    loss_G = loss_G_GAN + loss_NCE_both
    loss_G.backward()

    optimizer_F = torch.optim.Adam(patchMLP.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    scheulder_F = get_scheduler(optimizer_F, args)

    return optimizer_F, scheulder_F


if __name__ == "__main__":
    main()