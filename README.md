# A Pytorch Implementation of REDNet

“A Novel Recurrent Encoder-Decoder Structure for Large-Scale Multi-view Stereo Reconstruction from An Open Aerial Dataset” (CVPR 2020)

The proposed network was trained and tested on a single NVIDIA TITAN RTX 2080Ti (24G).

This project is based on the implementation of MVSNet-pytorch. Thank the author for providing the source code (https://github.com/xy-guo/MVSNet_pytorch)

## How to Use

### Environment
* python 3.7 (Anaconda)
* pytorch 1.1.0

## Data Preparation
1. Download the WHU MVS dataset.  http://gpcv.whu.edu.cn/data/WHU_dataset/WHU_MVS_dataset.zip. <br/>
                (The link in baidu disk: https://pan.baidu.com/s/1aGHFfW26Q8T4oCVa2tPZYQ code：91ae)
3. Unzip the dataset to the ```WHU_MVS_dataset``` folder. <br/>


## Train
1. In ```train.py```, set ```mode``` to ```train```, set ```model``` to ```rednet```<br/>
2. In ```train.py```, set ```trainpath``` to your train data path ```YOUR_PATH/WHU_MVS_dataset/train```,
                      set ```testpath``` to your train data path ```YOUR_PATH/WHU_MVS_dataset/test``` <br/>
2. Train REDNet (TITAN RTX 2080Ti 24G):<br/>
```
python train.py
```


## Test
1. In ```train.py```, set ```testpath``` to your train data path ```YOUR_PATH/WHU_MVS_dataset/test```,
   set ```loadckpt``` to your model path ```./checkpoints/whu_rednet/model_000005.ckpt```, set depth sample number ```numdepth```. <br/>
2. Run REDNet：<br/>
```
python train.py 
```
The test outputs are stored in ```YOUR_PATH/WHU_MVS_dataset/test/depths_rednet/```, including depth map ```XXX_init.pfm```, probability map ```XXX_prob.pfm```, scaled images ```XXX.jpg``` and camera parameters ```XXX.txt```. <br/>
