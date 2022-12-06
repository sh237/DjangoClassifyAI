#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

import requests
import zipfile
import urllib.request
import shutil
import numpy as np
import h5py
import json
import time
import torch
from torch import nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from  typing import Tuple
from image2text.ml_models import Encoder, DecoderWithAttention , Attention

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ImageClassifyProject.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
