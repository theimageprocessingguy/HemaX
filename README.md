
# Pathologist-Like Explanations Unveiled: an Explainable Deep Learning System for White Blood Cell Classification

This is the official PyTorch training code for HemaX used for training on LeukoX dataset. HemaX is trained on PyTorch version 1.13.1 with cuda enabled and torchvision version 0.14.1.


## Data Preparation

For training on custom dataset, training and validation directory must be created. A demo of the folder structure is shown below.
``` 
train
 	/inputs/
 	/bounding_boxes/
 	/instances/
 	/classes
```
**inputs** contains original images, **bounding_boxes** contains bounding box coordinates of the form ```[X, Y, W, H]``` where X = coordinate of the left of the bounding box, Y = coordinate of the top of the bounding box, W = width of the bounding box, H = height of the bounging box. **instances** contains segmentation mask of nucleus and cytoplasm of each image as separate channels, and **classes** contains class labels. Similarly validation directory has to be created. All the files inside these four subdirectory must be of **.pt** extension.
```train.txt``` and ```val.txt``` contain relative path to the images of the  **inputs** folder for training and validation respectively. The .txt files can be created by adding paths to ```list_img.py``` and executing it.
Test set should also be formatted in this way.  

## Execution

To start training, execute ```main.py``` with the desired arguments:
``` 
python main.py --argument_name argument_value ....
```


