# Segression
A Gaussian Text Detector for Arbitrary Shapes


# To-Do list
- [ ] Modularize the testing code for square images.
- [ ] Test all dataset to fix the evaluation metrics. 
- [ ] Pre-training of the VGG, ResNet50 and ResNest50.
- [ ] Fine tuned on datasets (TotalText, CTW1500, MSRA-TD500, ICDAR 2015)
- [ ] Abilation for 1d, 2d and 3d gaussian projection. 

# Experimentaion Details

Augmentation used:
- random scale [0.3,1.0]
- random crop 512 x 512
- random mirror
- random rotation [-30,30].

| Dataset     |  Fine Tune   |    Optimizer   | Learning Rate  | Decay policy | Batch size  | Itearation |
|-------------|--------------|----------------|----------------|--------------|-------------|------------|
| Synthetic   |    N         |     Adam       |    1e-4        |Linear        |    4        |  1000000   |
| Total Text  |    Y         |     Adam       |    1e-4        |Linear        |    4        |  1000000   |
| CTW1500     |    Y         |     Adam       |    1e-4        |Linear        |    4        |  1000000   |
| MSRA-TD500  |    Y         |     Adam       |    1e-4        |Linear        |    4        |  1000000   |
| ICDAR 2015  |    Y         |     Adam       |    1e-4        |Linear        |    4        |  1000000   |


# Pre-trained models

| Backbone    |  Dataset     |    Download Link   | 
|-------------|--------------|--------------------|
|VGG-16       | Synthetic    |                    |
|VGG-16       | Total Text   |                    |
|VGG-16       | CTW1500      |                    |
|VGG-16       | MSRA-TD500   |                    |
|VGG-16       | ICDAR 2015   |                    |
|ResNet50     | Synthetic    |                    |
|ResNet50     | Total Text   |                    |
|ResNet50     | CTW1500      |                    |
|ResNet50     | MSRA-TD500   |                    |
|ResNet50     | ICDAR 2015   |                    |
|ResNest50    | Synthetic    |                    |
|ResNest50    | Total Text   |                    |
|ResNest50    | CTW1500      |                    |
|ResNest50    | MSRA-TD500   |                    |
|ResNest50    | ICDAR 2015   |                    |


# Training
```
$ bash ./config/BACKBONE/DATASET.sh
```
BACKBONE : VGG_Configs, ResNest_Configs
DATASET  : Synth,total_text,ctw1500,msrtd-500,icdar_2015 


# Testing

For multi-scale testing on square wrapped images
```
$ python multi_scale_testing_nonmodular_square.py
```
For multi-scale testing on rectangular images 
```
$ python testing_modular_rectangular.py
```

# Results

Total Text 

| Backbone    |  Image scale | Gaussian Threshold  | Segmentation Threshold| Precision /Recall/F1-score |
|-------------|--------------|---------------------|-----------------------|----------------------------|
| VGG-16      |  512         |                     |                       |                            |
| VGG-16      |  512+128     |                     |                       |                            |
| VGG-16      |  512-128     |                     |                       |                            |
| VGG-16      |  512+256     |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |

CTW1500

| Backbone    |  Image scale | Gaussian Threshold  | Segmentation Threshold| Precision /Recall/F1-score |
|-------------|--------------|---------------------|-----------------------|----------------------------|
| VGG-16      |  512         |                     |                       |                            |
| VGG-16      |  512+128     |                     |                       |                            |
| VGG-16      |  512-128     |                     |                       |                            |
| VGG-16      |  512+256     |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |

MSRA-TD500

| Backbone    |  Image scale | Gaussian Threshold  | Segmentation Threshold| Precision /Recall/F1-score |
|-------------|--------------|---------------------|-----------------------|----------------------------|
| VGG-16      |  512         |                     |                       |                            |
| VGG-16      |  512+128     |                     |                       |                            |
| VGG-16      |  512-128     |                     |                       |                            |
| VGG-16      |  512+256     |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |

ICDAR 2015

| Backbone    |  Image scale | Gaussian Threshold  | Segmentation Threshold| Precision /Recall/F1-score |
|-------------|--------------|---------------------|-----------------------|----------------------------|
| VGG-16      |  512         |                     |                       |                            |
| VGG-16      |  512+128     |                     |                       |                            |
| VGG-16      |  512-128     |                     |                       |                            |
| VGG-16      |  512+256     |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |
| VGG-16      |              |                     |                       |                            |


