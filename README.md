# LGPNet
Our paper: "[Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy](https://ieeexplore.ieee.org/abstract/document/9627698)" has been published on <font size=5>**IEEE Transactions on Geoscience and Remote Sensing**</font> by Tongfei Liu, Maoguo Gong, Di Lu, Qingfu Zhang, Hanhong Zheng, Fenlong Jiang, and Mingyang Zhang.  

## Result Preview  
![Example results of our proposed LGPNet and the other methods on the WHUCD dataset: (a) T1- time images, (b) T2-time images, (c) Ground truth map, and (f) Our proposed LGPNet. (Notation: red color, cyan color, white color, and black color denote missed detection pixels, false detection pixels, correct detection changed pixels, correct detection unchanged pixels, respectively.)](https://github.com/TongfeiLiu/LGPNet/blob/main/Preview.png)

## Model Download Link
Link: https://pan.baidu.com/s/15_gvp9seONXpHK90LDJN0Q  
Password：yv9e

## Requirements
>python=3.7.10  
pytorch=1.9  
opencv-python=4.1.0.25  
scikit-image=0.14.2  
scikit-learn=0.24.1  
tqdm  

## Usage
### Train
1. Load the pretrain model path  
2. Load the train and test(val) data path  
python BCD_train.py  

### Test
1. Load the model path  
2. Load the test data path  
python BCD_test.py  

### Example(WHU)
**BCD_train.py** 
```
data_path = "./samples/WHU/train"  
epochs=110, batch_size=4, lr=0.0001, ModelName='DPN_Inria', is_Transfer= True  
BFENet.load_state_dict(torch.load('Pretrain_BFE_'+ModelName+'_model_epoch75_mIoU_89.657089.pth', map_location=device))  
```
run: python BCD_train.py  


**BCD_test.py**  
```
BFENet.load_state_dict(torch.load('BestmIoU_BFE_DPN_epoch91_mIoU_91.864527.pth', map_location=device))
BCDNet.load_state_dict(torch.load('BestmIoU_BCD_DPN_epoch91_mIoU_91.864527.pth', map_location=device))

tests1_path = glob.glob('./samples/WHU/test/image1/*.tif')  
tests2_path = glob.glob('./samples/WHU/test/image2/*.tif')  
label_path = glob.glob('./samples/WHU/test/label/*.tif')  
```
run: python BCD_test.py

**Note**  
We recommend importing the complete data set before executing training, otherwise, an error will be reported during the training process (Error: the denominator cannot be zero in the evaluation index calculation). 

### Get results (Visual and Quantitative)
**Visual result:** ./samples/WHU/test/results  
**Quantitative result:** ./test_acc.txt   

## Citation
If you find our work useful for your research, please consider citing our paper:  
```
@article{liu2021building,  
  title={Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy},  
  author={Liu, Tongfei and Gong, Maoguo and Lu, Di and Zhang, Qingfu and Zheng, Hanhong and Jiang, Fenlong and Zhang, Mingyang},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  year={2021},  
  pages={1-17},  
  doi={10.1109/TGRS.2021.3130940},  
  publisher={IEEE}  
}  
```

## Acknowledgement
This code is heavily borrowed from the PSPNet[1], PANet[2], DANet[3], etc. We are very grateful for the contributions of these papers and related codes. In addtion, we are also very grateful for the outstanding contributions of the publicly available datasets (WHU and LEVIR) of the papers [4] and [5].  
```
[1] Zhao H, Shi J, Qi X, et al. Pyramid scene parsing network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2881-2890.  
[2] Li H, Xiong P, An J, et al. Pyramid attention network for semantic segmentation[J]. arXiv preprint arXiv:1805.10180, 2018.  
[3] Fu J, Liu J, Tian H, et al. Dual attention network for scene segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3146-3154.  
[4] Ji S, Wei S, Lu M. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set[J]. IEEE Transactions on Geoscience and Remote Sensing, 2018, 57(1): 574-586.  
[5] Chen H, Shi Z. A spatial-temporal attention-based method and a new dataset for remote sensing image change detection[J]. Remote Sensing, 2020, 12(10): 1662.  
```

## Contact us 
If you have any problme when running the code, please do not hesitate to contact us. Thanks.  
E-mail: liutongfei_home@hotmail.com  
Date: Nov 7, 2021  
