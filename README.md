# Colorize-black-and-white-image_in-python-
Welcome to the Colorizing Black and White Images project! This Python-based project aims to bring life to black and white photographs by automatically adding vibrant colors to them.


# Overview
Image colorization is the process of taking an input grayscale (black and white) image and then producing an output colorized image that represents the semantic colors and tones of the input (for example, an ocean on a clear sunny day must be plausibly “blue” — it can’t be colored “hot pink” by the model).


**Here is a photo of Che guevara from 60's colorized:**




## Technical Aspect ##
The technique we’ll be covering here today is from Zhang et al.’s 2016 ECCV paper, Colorful Image Colorization. Developed at the University of California, Berkeley by Richard Zhang, Phillip Isola, and Alexei A. Efros.

Previous approaches to black and white image colorization relied on manual human annotation and often produced desaturated results that were not “believable” as true colorizations.

Zhang et al. decided to attack the problem of image colorization by using Convolutional Neural Networks to “hallucinate” what an input grayscale image would look like when colorized.

To train the network Zhang et al. started with the ImageNet dataset and converted all images from the RGB color space to the Lab color space.

Similar to the RGB color space, the Lab color space has three channels. But unlike the RGB color space, Lab encodes color information differently:

The L channel encodes lightness intensity only
The a channel encodes green-red.
And the b channel encodes blue-yellow.
As explained in the original paper, the authors, embraced the underlying uncertainty of the problem by posing it as a classification task using class-rebalancing at training time to increase the diversity of colors in the result. The Artificial Intelligent (AI) approach is implemented as a feed-forward pass in a CNN (“Convolutional Neural Network”) at test time and is trained on over a million color images.

The color photos were decomposed using Lab model and “L channel” is used as an input feature and “a and b channels” as classification labels as shown in below diagram.



The trained model (that is available publically and in models folder of this repo or download it by clicking here), we can use it to colorize a new B&W photo, where this photo will be the input of the model or the component “L”. The output of the model will be the other components “a” and “b”, that once added to the original “L”, will return a full colorized image.

## Installation And Run ##
1.The Code is written in Python 3.7. If you don't have Python installed you can find it here. If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

pip install -r requirements.txt

## Run the file with:
 streamlit run app.py
