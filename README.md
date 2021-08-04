## COVID-CS dataset and JCS model
Official repository of the paper: JCS: An explainable COVID-19 diagnosis system by joint classification and segmentation,
IEEE Transactions on Image Processing (TIP) 2021.

This repository contains:

* COVID-CS dataset data.
* **Training** and testing code of our JCS model.

This paper has been accepted and published in [IEEE Transactions on Image Processing (TIP) 2021](https://ieeexplore.ieee.org/document/9357961).

If you need any (special) help for the code and data, do not hesitate to leave issues in this repository.

Your can also directly contact me via e-mail: wuyuhuan (at) mail.nankai (dot) edu.cn


### Method

#### Requirements

A computer that should have **PyTorch >= 0.4.1 and CUDA**

It should have no big differences running on PyTorch 0.4.1 ~ 1.7.

#### Testing

* **Segmentation Test**

Please download the computed intermediate features at first: [OneDrive](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EfiCUqJ0oABAjQs5aHC-IScBmTIIaur_qV8Ldt2366JXPA?e=tvFhDV). Download the annotation files: [Google Drive](https://drive.google.com/file/d/1U489DgHNqlwLJ9VZa6qssf65SV9F45jc/view?usp=sharing).
Extract them into `./data/COVID-CS/` folder. Put features under `data/COVID-CS/feats_joint_pretrained` directory.
Download [the annotation files](https://drive.google.com/file/d/1U489DgHNqlwLJ9VZa6qssf65SV9F45jc/view?usp=sharing), and put them under the `./data/COVID-CS/` folder.

Then, download the JCS model weights (joint.pth): [Google Drive](https://drive.google.com/file/d/1V1EKXL4gFAH6ZtFRcmUv9-aI0sc5e9Ga/view). Put it into the `model_zoo/` folder.

After finishing the above steps, `tools/test_joint.sh` can be directly used to compute the final covid-19 results, like this:

```
bash ./tools/test_joint.sh
```

You should get 66.61% mIoU computed by our CUDA-based evaluator. 

* **Classification Test**

We provide a demo that generates **explainable attention maps** of CT images, as shown in our paper.

Examples of CT images have been in the `examples/` folder. The following steps are to generate the corresponding attention maps:

* First, please download the model weights:  `res2net_cls.pth` which only is trained with image labels and `res2net_segloss.pth` which is trained with pixel-level annotations. 
Downloading urls are  [res2net_cls.pth](https://drive.google.com/file/d/1rhLLZoeCBYQ7XWpEppywdL3mODlsJn9k/view?usp=sharing) and [res2net_segloss.pth](https://drive.google.com/file/d/1B431SuffibX9tBueSeVVoOL9TThmvjIz/view?usp=sharing), respectively. Put the weights into `model_zoo/` folder.
* Second, run `PYTHONPATH=$(pwd):$PYTHONPATH python tools/gen_results.py`, results will be generated in the `results_pos` folder. You should get same result with the illustrated figure in our paper.

#### Training

* Train Segmentation Model with Your Own Data

Please place your full data folder (e.g. `COVID-Dataset`) under the `./data` folder.
Then, create `train.txt` and `test.txt` following this format:

````
train/1.jpg train/1.png 
train/2.jpg train/2.png 
````

Where `train/1.jpg` is the CT image filename and `train/1.png` is the name of the corresponding annotation image.
Each line includes one CT image with a corresponding annotation image.

Edit `./tools/train.sh`, replace `--data_dir ./data/xxxxx` with `--data_dir ./data/COVID-Dataset`.

If you do not have data, you can train with provided examples as described below.

* Training Segmentation Model: An Example

We provide a training script with training examples which are located in the `data` folder.

Before training our model, please make sure that your GPU memory is higher than `4GB` for a batchsize of 1 or `6GB` for a batchsize of 2.

Then, please download the ImageNet-pretrained VGG-16 model [5stages_vgg16_bn-6c64b313.pth](https://drive.google.com/file/d/1zgO9vMCDpj2J50EExa28S3nWDNGQe5WC/view?usp=sharing) and put the model weights under the `model_zoo/` folder.
After the above preparation, you can simply run the following command to train our model: 

`CUDA_VISIBLE_DEVICES=0 ./tools/train.sh`

You can also prepare your data following the format of the data in the `./data/` folder. 

Note: The pure source segmentation data of our COVID-CS dataset can not be released so far according to some policies. Therefore, we utilize COVID-19-CT100 dataset to show how to train our model simply.


#### Precomputed Results

Joint pretrained results (cls + seg) (Dice: 78.5%): [Google Drive](https://drive.google.com/file/d/1ISi9LeFNyBOxKbtKTg2QCcOKX3dbkdNS/view).

Single pretrained results (seg only) (Dice: 77.5%): [Google Drive](https://drive.google.com/file/d/1r3-OL2veeRrBCyoVJ7JcSzY2atQAVu4Z/view).

#### Pretrained Models

* Segmentation:
* Joint pretrained model (cls + seg): [Google Drive](https://drive.google.com/file/d/1V1EKXL4gFAH6ZtFRcmUv9-aI0sc5e9Ga/view).
* Single pretrained model (seg only): [Google Drive](https://drive.google.com/file/d/1iXD9n1LSR7_pyyU8xQd0kZVn0IAat3Aq/view).
* Classification:
* Pretrained model (trained only with image labels): [Google Drive](https://drive.google.com/file/d/1rhLLZoeCBYQ7XWpEppywdL3mODlsJn9k/view?usp=sharing).
* Pretrained model (trained with pixel-level annotation): [Google Drive](https://drive.google.com/file/d/1B431SuffibX9tBueSeVVoOL9TThmvjIz/view?usp=sharing).

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

#### COVID-CS Data for Segmentation

Only features are provided. To train or test your model, you should skip the first stage of the backbone.
Then you can load our provided features and finish the inference of other stages:

```python
$NAME.mat, which contains two variables: vgg_feats, res2net_feats
import scipy.io as sio
import torch
feats = sio.load_mat('$NAME.mat')
vgg_feats = torch.from_numpy(feats["vgg_feats"]).float() / 20 # get vgg_feats conv1_2 (of VGG-16)
res2net_feats = torch.from_numpy(feats["res2net_feats"]).float() / 20 # get res2net_feats conv1 (of Res2Net-101-v1b)
output = model(vgg_feats, res2net_feats) or model(vgg_feats) or model(res2net_feats) # model inference
```

As default, the CT images are of `512 * 512` size. So vgg_feats and res2net_feats are of `1 * 64 * 512 * 512` and `1 * 64 * 256 * 256` size, respectively. To save disk space, we also provide the half size version (`256 * 256` CT image input size).

* Segmentation annotation data, including train/test split txt files: [Google Drive](https://drive.google.com/file/d/1U489DgHNqlwLJ9VZa6qssf65SV9F45jc/view?usp=sharing)
* JCS pretrained features: [OneDrive, 21.7GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EfiCUqJ0oABAjQs5aHC-IScBmTIIaur_qV8Ldt2366JXPA?e=tvFhDV)
* JCS pretrained features (half feature size): [OneDrive, 5.7GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EXjDhKvCRdZKjutnjSujHWcB6Fkjx329ZJI6wesnQ07Tog?e=GNkXZf)
* ImageNet pretrained features: [OneDrive, 19.8GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/EY01kub68GJPmzJmht97EaYBvX03anlgGgIJSeSAtitSWw?e=U0Totb)
* ImageNet pretrained features (half feature size): [OneDrive, 5.5GB](https://mailnankaieducn-my.sharepoint.com/:u:/g/personal/wuyuhuan_mail_nankai_edu_cn/Ebux1iLP1rxPvQTD66Ssi0ABg3bJYae9gGZc2q-j7gmB-A?e=irzXFy)

#### COVID-CS Data for Classification

The data are quite large. If you need a copy of the full 144K data, please contact me via E-mail!

## Others

If you need special help, please leave issues or directly contact me via e-mail. Thanks!
