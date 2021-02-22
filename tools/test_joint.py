import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from IoUEval import IoUEval
from Models import joint_model as net
import scipy.io as sio

@torch.no_grad()
def test(args, model, image_list, label_list):
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    eval = IoUEval()

    for idx in range(len(image_list)):
        if not args.input_features:
            image = cv2.imread(image_list[idx])
            label = cv2.imread(label_list[idx], 0)
            label = label / 255

            # resize the image to 1024x512x3 as in previous papers
            img = cv2.resize(image, (args.width, args.height))
            img = img.astype(np.float32) / 255.
            img -= mean
            img /= std

            img = img[:,:, ::-1].copy()
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)
            img = Variable(img)

            label = torch.from_numpy(label).float().unsqueeze(0)

            if args.gpu:
                img = img.cuda()
                label = label.cuda()

            start_time = time.time()
            
            img_out = model(img)[:, 0, :, :].unsqueeze(dim=0)
        else:
            label = cv2.imread(label_list[idx], 0)
            label = label / 255
            label = torch.from_numpy(label).float().unsqueeze(0)
            
            basename = osp.basename(label_list[idx])[:-4]
            feats = sio.loadmat(args.features_dir + basename + '.mat')
            feats_vgg = torch.from_numpy(feats['vgg_feats']).float() / 20.
            feats_res2net = torch.from_numpy(feats['res2net_feats']).float() / 20.
            
            
            if args.gpu:
                feats_vgg = feats_vgg.cuda()
                feats_res2net = feats_res2net.cuda()
                label = label.cuda()
            
            start_time = time.time()
            img_out = model(feats_vgg, res2net_features=feats_res2net)[:, 0, :, :].unsqueeze(dim=0)

            
        torch.cuda.synchronize()
        diff_time = time.time() - start_time
        print('\r Processing for {}/{} takes {:.3f}s per image'.format(idx, len(image_list), diff_time), end='')

        img_out = F.interpolate(img_out, size=label.shape[1:], mode='bilinear', align_corners=False)
        
        eval.add_batch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

        covid_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        
        cv2.imwrite(osp.join(args.savedir, osp.basename(image_list[idx])[:-4] + '.png'), covid_map)

    F_beta, MAE = eval.get_metric()
    print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))


def main(args):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, args.file_list)) as fid:
        for line in fid:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + line_arr[1].strip())

    model = net.JCS(input_features=args.input_features)
    if not osp.isfile(args.pretrained):
        print('Pretrained model file does not exist...')
        exit(-1)

    state_dict = torch.load(args.pretrained)
    model.load_state_dict(state_dict)
    #model = model.module

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    if not osp.isdir(args.savedir):
        os.makedirs(args.savedir)

    test(args, model, image_list, label_list)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data/COVID-CS", help='Data directory')
    parser.add_argument('--file_list', default="test.txt", help='Data directory')
    parser.add_argument('--width', type=int, default=512, help='Width of CT image')
    parser.add_argument('--height', type=int, default=512, help='Height of CT image')
    parser.add_argument('--savedir', default='./outputs/joint_model_result/', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default="model_zoo/joint.pth", help='Pretrained model')
    parser.add_argument('--input_features', type=int, default=0, help='whether directly input features')
    parser.add_argument('--features_dir', type=str, default='data/COVID-CS/feats_joint_pretrained/')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
