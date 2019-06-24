## Classification and Quantification of Retinal Cysts in OCT B-Scans: Efficacy of Machine Learning Methods
EMBC 2019

The automatic segmentation of fluid spaces in optical coherence tomography (OCT) imaging facilitates clinically relevant quantification and monitoring of eye disorders over time. Eyes with florid disease are particularly challenging to
segment, as the anatomy is often highly distorted from normal. In this context, we propose an end-to-end machine learning
method consisting of near perfect detection of retinal fluid using random forest classifier and an efficient DeepLab algorithm for quantification and labeling of the target fluid compartments. In particular, we achieve an average Dice score of 86.23% with reference to manual delineations made by a trained expert.

### Setup

Clone the repository and install dependencies from requirements.txt
```bash
git clone https://github.com/VamshiTeja/OCT-B-Scans-Segmentation
cd smdl
pip install -r requirements.txt
```

### Run

For classification:

```bash
cd classification
python binary_classification.py
```

For segmentation:

```bash

The only thing you have to do to get started is set up the folders in the following structure:

├── "dataset_name"
|   ├── train
|   ├── train_labels
|   ├── val
|   ├── val_labels
|   ├── test
|   ├── test_labels
```

Then you can simply run main.py! Check out the optional command line arguments:

```bash
python main.py [-h] [--num_epochs NUM_EPOCHS]
		[--mode MODE]
                [--image IMAGE][--class_balancing CLASS_BALANCING]
                [--continue_training CONTINUE_TRAINING] [--dataset DATASET]
                [--crop_height CROP_HEIGHT] [--crop_width CROP_WIDTH]
                [--batch_size BATCH_SIZE] [--num_val_images NUM_VAL_IMAGES]
                [--h_flip H_FLIP] [--v_flip V_FLIP] [--brightness BRIGHTNESS]
                [--rotation ROTATION] [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of epochs to train for
  --image IMAGE         The image you want to predict on. Only valid in
                        "predict" mode.
  --class_balancing CLASS_BALANCING
			Whether to use median frequency class weights to balance the classes in the loss
  --continue_training CONTINUE_TRAINING
                        Whether to continue training from a checkpoint
  --dataset DATASET     Dataset you are using.
  --crop_height CROP_HEIGHT
                        Height of cropped input image to network
  --crop_width CROP_WIDTH
                        Width of cropped input image to network
  --batch_size BATCH_SIZE
                        Number of images in each batch
  --num_val_images NUM_VAL_IMAGES
                        The number of images to used for validations
  --h_flip H_FLIP       Whether to randomly flip the image horizontally for
                        data augmentation
  --v_flip V_FLIP       Whether to randomly flip the image vertically for data
                        augmentation
  --brightness BRIGHTNESS
                        Whether to randomly change the image brightness for
                        data augmentation. Specifies the max bightness change
                        as a factor between 0.0 and 1.0. For example, 0.1
                        represents a max brightness change of 10% (+-).
  --rotation ROTATION   Whether to randomly rotate the image for data
                        augmentation. Specifies the max rotation angle in
                        degrees.
  --model MODEL         The model you are using.
```


### Acknowledgement

Thanks to [GeorgeSeif](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) for providing parts of the code!


### Citation

