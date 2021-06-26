import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from simclr_dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from simclr_model import FeaturesModel as Model
from test import validation

from byol_pytorch import BYOL
from imgaug import augmenters as iaa
import imgaug as ia

from tqdm import tqdm
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    
    ia.seed(1)
    image_transforms = iaa.Sequential([iaa.SomeOf((1, 5), 
                          [iaa.LinearContrast((0.5, 1.0)),
                          iaa.GaussianBlur((0.5, 1.5)),
                          iaa.Crop(percent=((0, 0.4),(0, 0),(0, 0.4),(0, 0.0)), keep_size=True),
                          iaa.Crop(percent=((0, 0.0),(0, 0.02),(0, 0),(0, 0.02)), keep_size=True),
                          iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                          iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'),
                          iaa.PerspectiveTransform(scale=(0.01, 0.02))],
                          random_order=True)])

    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, image_transforms=image_transforms)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    image_transforms = iaa.Sequential([iaa.SomeOf((1, 5), 
                          [iaa.LinearContrast((0.5, 1.0)),
                          iaa.GaussianBlur((0.5, 1.5)),
                          iaa.Crop(percent=((0, 0.4),(0, 0),(0, 0.4),(0, 0.0)), keep_size=True),
                          iaa.Crop(percent=((0, 0.0),(0, 0.02),(0, 0),(0, 0.02)), keep_size=True),
                          iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.5)),
                          iaa.PiecewiseAffine(scale=(0.02, 0.03), mode='edge'),
                          iaa.PerspectiveTransform(scale=(0.01, 0.02))],
                          random_order=True)])

    byol_learner = BYOL(
        model,
        image_size=(32,100),
        hidden_layer=-1,
        channels=1,
        augment_fn=image_transforms,
        augmented=True)

    print(byol_learner)

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, byol_learner.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))


    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps, weight_decay=opt.weight_decay)
    print("Optimizer:")
    print(optimizer)

    
    

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    #LR Scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.6*opt.num_iter), int(0.8*opt.num_iter)], last_epoch=start_iter-1, gamma=0.1)


    best_loss = None
    iteration = start_iter
    print(device)
    loss_avg = Averager()
    valid_loss_avg = Averager()
    # kl_loss_avg = Averager()
    # kl_loss = torch.nn.KLDivLoss()

    while(True):
        # train part
        for i in tqdm(range(opt.valInterval)):
            image_tensors, _ = train_dataset.get_batch()
            image = image_tensors.to(device)

            optimizer.zero_grad()

            loss = byol_learner(image)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            scheduler.step()
            byol_learner.update_moving_average()

            loss_avg.add(loss)

            if iteration==0:
                print("Epoch {:06d} Loss: {:.04f}".format(iteration, loss_avg.val()))

            iteration += 1

        
        byol_learner.eval()
        model.eval()
        with torch.no_grad():
            for image_tensors, _ in valid_loader:
                image = image_tensors.to(device)
                val_loss = byol_learner(image)
                valid_loss_avg.add(val_loss)

                # features = model(image)
                # features = features.view(-1, 26, features.shape[1])

                # kl_div = kl_loss(features[:int(features.shape[0]/2)], features[int(features.shape[0]/2):])
                # kl_loss_avg.add(kl_div)
        model.train()
        byol_learner.train()

        with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
            log.write("Iteration {:06d} Loss: {:.06f} Val loss: {:06f}".format(iteration, loss_avg.val(), valid_loss_avg.val()) + '\n')

        print("Iteration {:06d} Loss: {:.06f} Val loss: {:06f}".format(iteration, loss_avg.val(), valid_loss_avg.val()))
        if best_loss is None:
            best_loss = valid_loss_avg.val()
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')
        elif best_loss > valid_loss_avg.val():
            best_loss = valid_loss_avg.val()
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        scheduler.step()
        loss_avg.reset()
        valid_loss_avg.reset()

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')
            sys.exit()
        # iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--final_layer', type=int, default=128, help='final layer hidden state')
    parser.add_argument('--FinalLayer', action='store_true', help='Use a final projection layer')
    parser.add_argument('--weight_decay', type=float, default=10e-4, help='Weight decay')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-BYOL'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
