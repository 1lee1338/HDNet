import argparse
import time
import datetime
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from torch.utils.tensorboard import SummaryWriter
import random
from core.utils.score_1 import Evaluator_1

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Building Extraction Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='u_net',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus','hdnet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--name', type=str, default='-',
                        help='special option')
    parser.add_argument('--dataset', type=str, default='whu_satellite',
                        choices=['whu_satellite','whu_aerial','inria'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--aux-weight', type=float, default=20,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--continue_resume', type=bool, default=True,
                        help='training after previous checkpoint')
    parser.add_argument('--save-dir', default='../runs/models/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=100,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'whu_aerial': 200,
            'whu_satellite':200,
            'inria': 200,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'whu_aerial': 0.01,
            'whu_satellite': 0.01,
            'inria': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    #logger
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.model, args.backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()

class Trainer(object):
    def __init__(self, args,save_directory):
        self.args = args
        self.device = torch.device(args.device)
        self.save_directory = save_directory

        # dataset and dataloader
        data_kwargs = {'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            local_rank=args.local_rank,BatchNorm2d=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        if args.continue_resume:
            model_state_file = os.path.join(self.save_directory, 'checkpoint.pth.tar')
            if os.path.isfile(model_state_file):
                checkpoint = torch.load(model_state_file,map_location=lambda storage, loc:storage)
                self.best_pred = checkpoint['best']
                self.last_iter = checkpoint['iteration']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint (iteration {})".format(checkpoint['iteration']))

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)

        # dataload
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)


        # create criterion
        self.criterion = get_segmentation_loss(aux_weight=args.aux_weight, ignore_index=-1,local_rank=args.local_rank).to(self.device)

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        params_list.append({'params': self.model.parameters(), 'lr': args.lr})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = Evaluator_1(train_dataset.NUM_CLASS)

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations = {:d}'.format(epochs, max_iters))

        self.model.train()

        for iteration, (images, targets, _) in enumerate(self.train_loader):
            iteration = iteration + 1

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs,  m_c,m_b = self.model(images)
            loss_dict, loss_seg_dict, loss_edge_dict= self.criterion(outputs,  m_c,m_b, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_dict_reduced_seg = reduce_loss_dict(loss_seg_dict)
            losses_reduced_seg = sum(loss for loss in loss_dict_reduced_seg.values())
            loss_dict_reduced_edge = reduce_loss_dict(loss_edge_dict)
            losses_reduced_edge = sum(loss for loss in loss_dict_reduced_edge.values())

            writer.add_scalar('Train/Loss',losses, global_step=iteration, walltime= None)
            writer.add_scalar('Train/Redeced Loss',losses_reduced, global_step=iteration, walltime= None)
            writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], global_step=iteration, walltime=None)
            writer.add_scalar('Train/loss_seg', losses_reduced_seg, global_step=iteration, walltime=None)
            writer.add_scalar('Train/loss_edge', losses_reduced_edge, global_step=iteration, walltime=None)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                logger.info(
                    "Epoch:{:d} Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Loss_seg: {:.4f} || Loss_edge: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        int(iteration/self.args.iters_per_epoch), iteration, max_iters, self.optimizer.param_groups[0]['lr'],
                        losses_reduced.item(),losses_reduced_seg.item(),losses_reduced_edge.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, iteration,self.save_directory,is_best=False,optimizer=self.optimizer )

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(iteration)
                self.model.train()

        save_checkpoint(self.model, self.args, iteration,self.save_directory,is_best=False,is_final=True)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))
        logger.info(
            "Best model: Recall: {:.3f}, Precision: {:.3f},OA: {:.3f}, IoU: {:.3f}, F1: {:.3f}, lr: {:.3f}".format(
                self.best_Recall,self.best_Precision,self.best_OA,self.best_IoU,self.best_F1,self.optimizer.param_groups[0]['lr']))

    def validation(self,iteration):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()

        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs,m_c,m_b= model(image)

            loss_dict_val,_,_ = self.criterion(outputs, m_c,m_b ,target)
            losses_val = sum(loss for loss in loss_dict_val.values())
            self.metric.add_batch(target, outputs[0])

        F1, IoU, Recall, Precision, OA = self.metric.f1_miou_test()
        F1 = F1[1]
        IoU = IoU[1]
        Recall = Recall[1]
        Precision = Precision[1]

        writer.add_scalar('Val/F1', F1, global_step=iteration, walltime=None)
        writer.add_scalar('Val/Recall', Recall, global_step=iteration, walltime=None)
        writer.add_scalar('Val/Precision', Precision, global_step=iteration, walltime=None)
        writer.add_scalar('Val/IoU', IoU, global_step=iteration, walltime=None)
        writer.add_scalar('Val/OA', OA, global_step=iteration, walltime=None)
        writer.add_scalar('Val/loss', losses_val, global_step=iteration,  walltime=None)
        new_pred = IoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            save_checkpoint(self.model, self.args,iteration,self.save_directory,is_best,self.best_IoU)
        logger.info(
            "Total Validation Overall Acc: {:.4f}, F1: {:.4f}, Iou: {:.4f}, Recall: {:.4f},  Precision: {:.4f}, best:{:.4f}".format(
                OA, F1, IoU, Recall, Precision, self.best_pred))
        synchronize()

def save_checkpoint(model, args,iteration,save_directory, is_best=False,best_miou=0,optimizer=False,is_final=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(save_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)
    elif is_final:
        filename = '{}_{}_{}_final.pth'.format(args.model, args.backbone, args.dataset)
        filename = os.path.join(directory, filename)
        torch.save(model.state_dict(), filename)
    else :
        logger.info('=> saving checkpoint to {}'.format(
            directory + 'checkpoint.pth.tar'))
        torch.save({
            'iteration': iteration,
            'best': best_miou,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(directory, 'checkpoint.pth.tar'))


if __name__ == '__main__':
    dt = datetime.datetime.now().strftime('%Y-%m-%d-%H:%S-%P')
    args = parse_args()

    save_directory = os.path.join(args.save_dir, args.dataset, args.model, args.backbone,args.name)
    #tensorboard
    tensorboard_dic = os.path.join(save_directory,'tensorboard')
    m = os.path.isfile(tensorboard_dic)
    if os.path.exists(tensorboard_dic) is False:
        os.makedirs(tensorboard_dic)
    writer = SummaryWriter(tensorboard_dic  + '-' +dt)

    num_gpus = 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1

    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus
    #loger
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_directory = os.path.join(args.log_dir, args.dataset, args.model, args.backbone,args.name)
    logger = setup_logger("semantic_segmentation", log_directory, get_rank(), filename='{}_{}_{}_{}log.txt'.format(
        args.model, args.backbone, args.dataset,time_str))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args,save_directory)
    trainer.train()
    torch.cuda.empty_cache()