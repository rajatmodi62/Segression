# Segression
A Gaussian Text Detector for Arbitrary Shapes

# To-Do list
- [ ] Convert 1D, 2D to non vectorized using non vectorized
- [x] Modularize the testing code for square images.
- [x] Test all dataset to fix the evaluation metrics.
- [ ] Pre-training of the VGG, ResNet50 and ResNest50.
- [x] Fine tuned on datasets (TotalText, CTW1500, ICDAR 2015)
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
| ICDAR 2015  |    Y         |     Adam       |    1e-4        |Linear        |    4        |  1000000   |


# Pre-trained models

| Backbone    |  Dataset     |    Download Link                                                                              |
|-------------|--------------|-----------------------------------------------------------------------------------------------|
|VGG-16       | Synthetic    | (download)[https://drive.google.com/file/d/1u8lcuyE7poKJfEWnQlm3oQHlYA1lZ-62/view?usp=sharing]|
|VGG-16       | Total Text   |                                                                                               |
|VGG-16       | CTW1500      |                                                                                               |
|VGG-16       | ICDAR 2015   |                                                                                               |
|ResNet50     | Synthetic    |                                                                                               |
|ResNet50     | Total Text   |                                                                                               |
|ResNet50     | CTW1500      |                                                                                               |
|ResNet50     | ICDAR 2015   |                                                                                               |
|ResNest50    | Synthetic    |                                                                                               |
|ResNest50    | Total Text   |                                                                                               |
|ResNest50    | CTW1500      |                                                                                               |
|ResNest50    | ICDAR 2015   |                                                                                               |


# Training
```
$ bash ./config/BACKBONE/DATASET.sh
```
BACKBONE : VGG_Configs, ResNest_Configs
DATASET  : Synth,total_text,ctw1500,msrtd-500,icdar_2015

# Dataset
(Link)[]  to donwload the Dataset. Unzip the folder and place inside the data folder.

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
