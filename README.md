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

To be updated

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

Related data are still preparing. We will release the data soon.
