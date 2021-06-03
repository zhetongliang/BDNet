# BDNet-Pytorch
A decoupled learning scheme for training a burst denoising network for real-world denoising.

[[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154.pdf)][[Supp](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154-supp.pdf)]

Note: we only provides the code for real-world noise removal. 
## Environment
* Pytorch == 1.2.0
* Cuda == 10.1
* Python == 3.7

## Datasets
### Training
* [Dynamic video dataset](https://drive.google.com/file/d/1GncC7zXElvaGMgVnxySz7OiZU7980rJ6/view?usp=sharing).
 This dataset corresponds to the _D<sub>d<sub>_ in the main paper which provides dynamic contents for alignment learnig.
* [Real-world static burst dataset](https://drive.google.com/file/d/1hNhKih9qhpFzErh1l9Sx1elq2N1CM408/view?usp=sharing).
This dataset corresponds to the _D<sub>s<sub>_ in the main paper which provides real-world noise for learning.
### Testing
* [Real-static](https://drive.google.com/file/d/1-iipNmD7HO2iFsvAPMovoNKyFh1JMF8-/view?usp=sharing).
* Real-dynamic (to be finished)

## Pretrained model
[link](https://connectpolyu-my.sharepoint.com/:u:/r/personal/16903705r_connect_polyu_hk/Documents/BDNet_release/pretrained_model/pretrained_model.pth?csf=1&web=1&e=ehsmpL)

## Usage
This project is based on the EDVR project from https://github.com/xinntao/EDVR/tree/old_version. For details, you may refer to that project. We will simplify the code in the future.
1. Download the training datasets, including [Dynamic video dataset](https://connectpolyu-my.sharepoint.com/:f:/r/personal/16903705r_connect_polyu_hk/Documents/BDNet_release/datasets/vimeo90k?csf=1&web=1&e=IoA2rw) and [Real-world static burst dataset](https://connectpolyu-my.sharepoint.com/:f:/r/personal/16903705r_connect_polyu_hk/Documents/BDNet_release/datasets/Real_train?csf=1&web=1&e=s4VLOc), and unzip them to the path `datasets`.
2. Download the testing sets. Currently we only have [Real-static](https://connectpolyu-my.sharepoint.com/:f:/r/personal/16903705r_connect_polyu_hk/Documents/BDNet_release/datasets/Real_static?csf=1&web=1&e=YXhF0A) and we will update Real-dynamic set in the future. Download them and unzip them to the path `datasets`.
3. For training, `python train_BDNet.py`
4. For testing, you may download the [pretrained models](https://connectpolyu-my.sharepoint.com/:u:/r/personal/16903705r_connect_polyu_hk/Documents/BDNet_release/pretrained_model/pretrained_model.pth?csf=1&web=1&e=ehsmpL) first and put it in `pretrained_model` folder. Then `python test_BDNet.py`

## Contact
Zhetong Liang <zhetong.liang@connect.polyu.hk>

## Citation
@inproceedings{liang2020bdnet,
  title={A Decoupled Learning Scheme for Real-worldBurst Denoising from Raw Images},
  author={Zhetong Liang, Shi Guo, Hong Gu, Huaqi Zhang, and Lei Zhang},
  booktitle={ECCV},
  year={2020}
}
