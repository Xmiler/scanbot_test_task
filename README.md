# Problem Statement
Install Jupyter, matplotlib and opencv-python, and open mrz-lab.ipynb

# Nov 19, 2019 - Version 1.1.0
Several experiments were done.

1. I changed architecture on the standard resnet18 one. It raises model's metrics on the current data sampling, but at the same time works worse on semi-real and real photos. It's common problem when we try to use too power CNN and don't have enough data (too simple data sampling strategy in our case). Commit: `9a23351`. 

2. I augmented data sampling strategy with color jittering, noise and geometry (rotation + perspective). Then I finetuned CNN from `9a23351` with reduced learning rate. It works better on semi-real sample (than v1.0.0) and starts to work on a real one.

Based on these observation I think the core impact on quality is a closing gap between real photo and synthetic one. One can see that false accepts [there](`dataset_demo.ipynb`) are objects that CNN didn't see before during training: stamp and hand.  

# Nov 19, 2019 - Version 1.0.0
This is initial version of the "<"-segmentation subtask. It introduces base implementation of data synthesis module (see `dataset_demo.ipynb`) and training procedure. 

There are several issues related to inefficient training. But that's enough to make couple of experiments I'd like to share with you.

I've trained UNet architecture with primitive backbone on 512x512 gray-scale input. It gives precision/recall ~95% (with threshold 0.5) and works on the photo attached by you. But it doesn't work on the more realistic one because of current data synthesis module doesn't support such variations. See `inference.ipynb`.

It seems that current bottleneck is the ability of synthetic data to coverage aspects of real image. Current neural network gives high quality using current synthetic data.

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

## Update
Based on feedback got through `ml-challenge@scanbot.io` at Oct 31th, 2019 I've decided to explain motivation of the current step more. 

Honestly I hadn't understood what exactly you were expecting as a result and I'll find out it separately. But there I decided to interpret task as a business one, i.e. what should I do to get a product version ready to launch on a real scenario.

So it isn't just yet another opensource that I tried to hook as a solution. But a start point for me as a person who hasn't faced with this area before. It gives me such information as 
* there are datasets in a academia that proved their worth for my concrete business case
* algorithm itself seems pretty simple both for training and inference - Unet architecture with utilizing both location of character and space symbol. Furthemore it has [published at CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.pdf).
* special data sampling practices

As it has `train.py` tool I can finetune this solution on my own data domain immediately. Or I can find architecture that fits preferable Accuracy/Speed trade-off.

Yeah, it's purely engineer side of the given task. That's the point.
