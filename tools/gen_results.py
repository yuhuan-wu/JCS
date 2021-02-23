import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
import torchvision.transforms as transforms
from PIL import Image
from Models.res2net import res2net101_v1b_26w_4s
import torch.utils.data as data
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            # print(name,x.shape)
            if name == 'avgpool':
                break
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.fc(output)
        return target_activations, output

def preprocess_image(img):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]) # 为了训练一致所以先resize再裁剪。如果不想裁剪需要重新训练不裁剪版本的model
    preprocessed_img = val_trans(img)
    preprocessed_img.unsqueeze_(0)
    # print("preprocessed_img", preprocessed_img.shape)
    crop_trans =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
    crop_img = crop_trans(img)
    input = Variable(preprocessed_img, requires_grad = True)
    return crop_img, input

def show_cam_on_image(img, mask, save_path):
    heatmap = cv2.applyColorMap(np.uint8(256*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        print("output",output.cpu().data.numpy())
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        self.model.fc.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output
    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[1,:,:,:]

        return output



class OnePatientDataset(data.Dataset):
    def __init__(self, image_path, pos=True):
        self.imgs = os.listdir(image_path)
        self.image_path = image_path
        if pos:
            self.label = 1
        else:
            self.label = 0

    def prep_image(self, img):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        val_trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]) # 为了训练一致所以先resize再裁剪
        preprocessed_img = val_trans(img)
        return preprocessed_img

    def __getitem__(self, index):#返回的是tensor
        img = pil_loader(os.path.join(self.image_path,self.imgs[index]))
        prep_img = self.prep_image(img)
        return prep_img, self.label

    def __len__(self):
        return len(self.imgs)



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='examples',
                        help='Input image path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch-size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--resume', type=str, default="model_zoo/res2net_segloss.pth",
                        help='pretrained model weight path')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def get_cam_for_one_patient(grad_cam, image_path, patient, saveto='cam_results'):
    imgs = os.listdir(os.path.join(image_path,patient))
    save_path = os.path.join(saveto, patient)
    # save_path = save_path.split('/')
    # if save_path[-1]=="":
    #     save_path = save_path[-2]
    # else:
    #     save_path = save_path[-1]
    print(save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for img_name in imgs:
        img = pil_loader(os.path.join(image_path, patient, img_name))
        crop_image, input = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = 1

        mask = grad_cam(input, target_index)

        show_cam_on_image(crop_image, mask, os.path.join(save_path, img_name))

def get_result_for_one_patient(model, image_path, pos=True):
    val_loader = torch.utils.data.DataLoader(
        OnePatientDataset(image_path, pos=pos),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = torch.softmax(model(input), dim=1)
            if i==0:
                vector = output
            else:
                vector = torch.cat((vector, output), dim=0)
    print(vector.t())
    # rule 1 # max win and get set threshold for images.
    _, pred = vector.topk(1, 1, True, True)
    ratio = torch.sum(pred)/vector.shape[0]
    if torch.sum(pred)>18:
        return 1
    else:
        return 0


if __name__ == '__main__':
    """ 
    Res2Net CAM FOR CT IMAGE.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model = res2net101_v1b_26w_4s(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2, bias=False)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint, strict=False)
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    grad_cam = GradCam(model = model, \
                    target_layer_names = ["layer4"], use_cuda=args.cuda)
    
    get_cam_for_one_patient(grad_cam, image_path='', patient = args.image_path, saveto='results_pos')

