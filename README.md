# LGPNet
our recent paper: "[Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy](https://ieeexplore.ieee.org/abstract/document/9627698)" has been published on IEEE Transactions on Geoscience and Remote Sensing.  

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

## Train
1. Load the pretrain model path  
2. Load the train and test(val) data path  
python BCD_train.py  
**Example:**  

## Test
1. Load the model path  
2. Load the test data path  
python BCD_test.py  
**Example:**  

## Get results (Visual and Quantitative)
Visual result: ./samples/test/result  
Quantitative result: ./test_acc.txt   

## Citation
If you find our work useful for your research, please consider citing our paper:  

>@article{liu2021building,  
  title={Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy},  
  author={Liu, Tongfei and Gong, Maoguo and Lu, Di and Zhang, Qingfu and Zheng, Hanhong and Jiang, Fenlong and Zhang, Mingyang},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  year={2021},  
  pages={1-17},  
  doi={10.1109/TGRS.2021.3130940},  
  publisher={IEEE}  
}  


## Acknowledgement
This code is heavily borrowed from the PSPNet[1], PANet[2], DANet[3], etc. We are very grateful for the contributions of these papers and related codes. 
[1] Zhao H, Shi J, Qi X, et al. Pyramid scene parsing network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2881-2890.  
[2] Li H, Xiong P, An J, et al. Pyramid attention network for semantic segmentation[J]. arXiv preprint arXiv:1805.10180, 2018.  
[3] Fu J, Liu J, Tian H, et al. Dual attention network for scene segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 3146-3154.  

```
    function fun(){
         echo "这是一句非常牛逼的代码";
    }
    fun();
```
