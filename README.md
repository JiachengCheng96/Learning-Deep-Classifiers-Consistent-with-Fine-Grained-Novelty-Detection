# Novelty Detection Consistent Classifier (NDCC)
This is the code for the paper: \
[Learning Deep Classifiers Consistent with Fine-Grained Novelty Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Learning_Deep_Classifiers_Consistent_With_Fine-Grained_Novelty_Detection_CVPR_2021_paper.html). \
Jiacheng Cheng and Nuno Vasconcelos. In CVPR 2021.

If you find this repo useful for your research, please cite this paper as: 
```
@InProceedings{Cheng_2021_CVPR,
    author    = {Cheng, Jiacheng and Vasconcelos, Nuno},
    title     = {Learning Deep Classifiers Consistent With Fine-Grained Novelty Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {1664-1673}
}
```

## Dependencies
We implement our methods in PyTorch on NVIDIA GPUs. The environment requirements are as bellow:
- PyTorch, version >= 1.4.0
- torchvision, version >= 0.5.0
- sklearn, version >= 0.23.2
- NumPy
- pandas


## Datasets
We evaluated NDCC on multiple fine-grained visual categorization datasets ([CUB-200-2010](http://www.vision.caltech.edu/visipedia/CUB-200.html), [StanfordDogs](http://vision.stanford.edu/aditya86/ImageNetDogs/), and [FounderType200](https://www.icst.pku.edu.cn/zlian/representa/cv017/index.htm)). If you use these datasets, please cite the corresponding papers.