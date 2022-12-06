# DjangoClassifyAI

"DjangoClassifyAI" is  is a web application that can perform inference tasks using images and text as input.

# DEMO

This application can perform three inference tasks: the first is to determine the model of a Nike sneaker, the second is to generate a description from an image, and the third is to automatically create a title for the text.

https://user-images.githubusercontent.com/78858054/205949982-3f423ff1-9da9-4407-af74-82145464f775.mov

You can perform inference tasks without logging in, but logging in allows you to save inference results as history.

# Features

You can learn how to perform inference in web applications using tensorflow and pytorch.
You can display diagrams created with matplotlib in Django.

# Requirement

* Django 4.1.3
* django-bootstrap5 22.2
* django-widget-tweaks 1.4.12
* easydict 1.10
* googletrans 4.0.0rc1
* h5py 3.7.0
* keras 2.11.0
* matplotlib 3.6.2
* nltk 3.7
* numpy 1.23.5
* Pillow 9.3.0
* scikit-image 0.19.3
* scipy 1.9.3
* sentencepiece 0.1.97
* tensorboard 2.11.0
* tensorboard-data-server 0.6.1
* tensorboard-plugin-wit 1.8.1
* tensorflow 2.11.0
* tensorflow-estimator 2.11.0
* tensorflow-io-gcs-filesystem 0.28.0
* tokenizers 0.13.2
* torch 1.13.0
* torchvision 0.14.0
* tqdm 4.64.1
* transformers 4.25.1
* typing_extensions 4.4.0



# Installation

Install with requirements.txt.

```bash
pip install -r requirements.txt
```

# Usage

The following commands are executed to make it work in localhost.

```bash
python manage.py createsuperuser
python manage.py runserver
```
You can access the admin site by creating a super user. Also, by logging in, the inference results can be saved.
In the Nike sneaker identification AI and the AI that generates text from images, you can upload images by clicking on the area where images can be uploaded, and then click on the "infer button" to move to the loading screen. By waiting from that state, the inference results are displayed. In the AI for titling sentences, the user enters the sentence in the designated frame and presses the "infer button" to move to the loading screen. Then, the user waits for the results of inference to be displayed.

# Note

Currently, only the inference results of the generation text creation AI can be stored.

# Author

* Shunya Nagashima
* Twitter : -
* Email : syun864297531@gmail.com

# License

"DjangoClassifyAI" is under [MIT license].

Thank you!
