import torch
import cv2
import os.path as osp
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, dataset, transform=None):
        self.transform = transform
        self.img_list = list()
        self.msk_list = list()
        with open(osp.join(data_dir, dataset + '.txt'), 'r') as lines:
            for line in lines:
                line_arr = line.split()
                self.img_list.append(osp.join(data_dir, line_arr[0].strip()))
                self.msk_list.append(osp.join(data_dir, line_arr[1].strip()))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx])
        label = cv2.imread(self.msk_list[idx], 0)
        if self.transform:
            [image, label] = self.transform(image, label)
        return image, label

    def get_img_info(self, idx):
        image = cv2.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
