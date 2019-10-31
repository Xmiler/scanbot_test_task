# Problem Statement
Install Jupyter, matplotlib and opencv-python, and open mrz-lab.ipynb

# Oct 31, 2019 - Version 0.0.0
## Overview
This is a trial version that doesn't solve a problem completely but aimed to demonstrate direction of investigation. It's based on opensource text detector (released by https://ocr.clova.ai/ :)) and seems it has not bad robustness (see below). If I understood correctly the objective is to detect MRZ text and I really haven't to detect `<` symbols.
If so I can get final solution

1) by applying some postprocessing based on a priori information about MRZ layout if I have it

2) or even by direct text recognition using [this one](https://github.com/clovaai/deep-text-recognition-benchmark) if I don't have

## Install
Download [weight](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) for third-party model
```
git submodule update --init thirg-party/CRAFT-pytorch
mkdir third-party/CRAFT-pytorch/resources
mv ~/Download/craft_mlt_25k.pth third-party/CRAFT-pytorch/resources
```
Follow installation instruction at `third-party/CRAFT-pytorch/README.md`. Issues I faced for:
1) `third-party/CRAFT-pytorch/requirements.txt`: `torch==0.4.1.post2` -> `torch==0.4.1`
2) `CUDA 9`

## Run
```
python third-party/CRAFT-pytorch/test.py --trained_model third-party/CRAFT-pytorch/resources/craft_mlt_25k.pth --test_folder demo_images/
```
One can add his own samples to `demo_images` and get predictions on these.

## Notes
1. Current neural network is sensitive to input resolution. I got a better results on images with original size. Low dimensional and bad camera conditions affect multiplicatively.
2. I attached photos of my personal document purposely for demonstration purposes. I'm sure you will use these with care. Thanks.
