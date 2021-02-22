## COVID-CS dataset and JCS model
Official repository of the paper: JCS: An explainable COVID-19 diagnosis system by joint classification and segmentation,
IEEE Transactions on Image Processing (TIP) 2021.

This repository contains:

* COVID-CS dataset data.
* Training and testing code of our JCS model.

This paper has been accepted and published in IEEE Transactions on Image Processing (TIP) 2021.

## Method

### Introduction

Recently, the coronavirus disease 2019 (COVID-19) has caused a pandemic disease in over 200 countries, influencing billions of humans. To control the infection, identifying and separating the infected people is the most crucial step. The main diagnostic tool is the Reverse Transcription Polymerase Chain Reaction (RT-PCR) test. Still, the sensitivity of the RT-PCR test is not high enough to effectively prevent the pandemic. The chest CT scan test provides a valuable complementary tool to the RT-PCR test, and it can identify the patients in the early-stage with high sensitivity. However, the chest CT scan test is usually time-consuming, requiring about 21.5 minutes per case. This paper develops a novel Joint Classification and Segmentation (JCS) system to perform real-time and explainable COVID-19 chest CT diagnosis. To train our JCS system, we construct a large scale COVID-19 Classification and Segmentation (COVID-CS) dataset, with 144,167 chest CT images of 400 COVID-19 patients and 350 uninfected cases. 3,855 chest CT images of 200 patients are annotated with fine-grained pixel-level labels of opacifications, which are increased attenuation of the lung parenchyma. We also have annotated lesion counts, opacification areas, and locations and thus benefit various diagnosis aspects. Extensive experiments demonstrate that the proposed JCS diagnosis system is very efficient for COVID-19 classification and segmentation. It obtains an average sensitivity of 95.0% and a specificity of 93.0% on the classification test set, and 78.5% Dice score on the segmentation test set of our COVID-CS dataset.

### Requirements

A computer that should have **PyTorch >= 0.4.1 and CUDA**

It should have no big differences running on PyTorch 0.4.1 ~ 1.7.

### Testing

Please download the computed intermediate features at first: [OneDrive](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EfiCUqJ0oABAjQs5aHC-IScBmTIIaur_qV8Ldt2366JXPA?e=tvFhDV)

Put features under `data/COVID-CS/feats_joint_pretrained` directory.
Then, `tools/test_joint.sh` can be used to directly compute the final covid-19 results:

```
bash ./tools/test_joint.sh
```

You should get 66.61% mIoU computed by our CUDA-based evaluator. 

### Training

To be updated...

### Precomputed Results

Joint pretrained results (cls + seg) (Dice: 78.5%): [Google Drive](https://drive.google.com/file/d/1ISi9LeFNyBOxKbtKTg2QCcOKX3dbkdNS/view).

Single pretrained results (seg only) (Dice: 77.5%): [Google Drive](https://drive.google.com/file/d/1r3-OL2veeRrBCyoVJ7JcSzY2atQAVu4Z/view).

### Pretrained Models

Joint pretrained model (cls + seg): [Google Drive](https://drive.google.com/file/d/1V1EKXL4gFAH6ZtFRcmUv9-aI0sc5e9Ga/view).

Single pretrained model (seg only): [Google Drive](https://drive.google.com/file/d/1iXD9n1LSR7_pyyU8xQd0kZVn0IAat3Aq/view).

## COVID-CS Dataset

Our COVID-CS dataset includes more than **144K** CT images from **400** COVID-19 patients and **350** healthy persons.

Among our dataset, 3,855 images of 200 positive cases are pixel-level annotated, 64,771 images of the other 200 positive cases are patient-level annotated, and the rest 75,541 images are from the 350 negative cases.


Due to our confidential agreement with the data provider, we cannot share the source CT images. 
Therefore, we provide **CNN features of CT images**. Currently, we provide the intermediate features.

We generate intermediate features with the JCS model or ImageNet-pretrained model.
More specifically, only features at the first stage are restored.
To save space, we quantize the features into uint8 format.
Taking the VGG-16-based backbone as the example:

```python
 x = vgg16.conv1_1(ct_images)
 features = vgg16.conv1_2(x) # We record features conv1_2 of VGG-16
 features = (features.cpu().numpy() * 20).astype(numpy.uint8) # To save space, we save features as the uint8 format.
 ### A coefficient 20 is to more effeciently utilize the space of uint8 variables
 
 save(features, save_path) # Saveing features
```

As JCS has two backbones, we save vgg_features and res2net_features for each CT image.

### COVID-CS Data for Segmentation

Only features are provided. To train or test your model, you should skip the first stage of the backbone.
Then you can load our provided features and finish the inference of other stages:

```
$NAME.mat, which contains two variables: vgg_feats, res2net_feats
import scipy.io as sio
import torch
feats = sio.load_mat('$NAME.mat')
vgg_feats = torch.from_numpy(feats[vgg_feats]).float() * 20 # get vgg_feats conv1_2 (of VGG-16)
res2net_feats = torch.from_numpy(feats[res2net_feats]).float() * 20 # get res2net_feats conv1 (of Res2Net-101-v1b)
output = model(vgg_feats, res2net_feats) or model(vgg_feats) or model(res2net_feats) # model inference
```

As default, the CT images are of $512\times 512$ size. So vgg_feats and res2net_feats are of $1\times 64 \times 512 \times 512$ and $1\times 64 \times 256 \times 256$ size, respectively. To save disk space, we also provide the half size version ($256 \times 256$ CT image input size).

* Segmentation annotation data, including train/test split txt files: [Google Drive](https://drive.google.com/file/d/1U489DgHNqlwLJ9VZa6qssf65SV9F45jc/view?usp=sharing)
* JCS pretrained features: [OneDrive, 21.7GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EfiCUqJ0oABAjQs5aHC-IScBmTIIaur_qV8Ldt2366JXPA?e=tvFhDV)
* JCS pretrained features (half feature size): [OneDrive, 5.7GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EXjDhKvCRdZKjutnjSujHWcB6Fkjx329ZJI6wesnQ07Tog?e=GNkXZf)
* ImageNet pretrained features (3855 images): [OneDrive, 19.8GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EY01kub68GJPmzJmht97EaYBvX03anlgGgIJSeSAtitSWw?e=U0Totb)
* ImageNet pretrained features: [OneDrive, 5.5GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/Ebux1iLP1rxPvQTD66Ssi0ABg3bJYae9gGZc2q-j7gmB-A?e=irzXFy)

### COVID-CS Data for Classification

Still preparing...


