from numpy.random import randint
from PIL import Image
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

import imgaug.augmenters as iaa

from .synthetic_card_image_generator import SyntheticCardImageGenerator, generate_random_text_image, generate_random_face_image


class SyntheticCardImageDataset(Dataset):
    transform = transforms.Compose([transforms.ToTensor()])

    def __init__(self, size, to_tensor=False, fake_epoche_size=1000, hardness=0):
        self._size = size
        self._to_tensor = to_tensor
        self._fake_epoch_size = fake_epoche_size

        assert hardness in [0, 1]
        self._trsf_extra = {'font_fill': 0,
                            'fill': 255,
                            'geom_augm': None,
                            'static_augm': None}
        if hardness == 1:
            self._trsf_extra['font_fill'] = randint(0, 50)
            self._trsf_extra['fill'] = randint(150, 256)
            self._trsf_extra['geom_augm'] = iaa.Sequential([iaa.Affine(rotate=(-1, 1), mode='symmetric'),
                                                            iaa.PerspectiveTransform(scale=0.005, mode='replicate')])
            self._trsf_extra['static_augm'] = iaa.Sequential([iaa.AdditivePoissonNoise(lam=3)])

    def __len__(self):
        return self._fake_epoch_size

    def __getitem__(self, idx):

        card_image = SyntheticCardImageGenerator(font_fill=self._trsf_extra['font_fill'],
                                                 fill=self._trsf_extra['fill'])

        for _ in range(25):
            text_image = generate_random_text_image(randint(5, 20),
                                                    font_fill=self._trsf_extra['font_fill'],
                                                    fill=self._trsf_extra['fill'])
            card_image.put_fragment(text_image)

        face_image = generate_random_face_image(randint(150, 350))

        card_image.put_fragment(face_image)

        card_image.draw_pattern_texts()

        card_image.apply_square_and_zooming()

        card_image.resize(self._size)

        image, image_gt = card_image.image, card_image.image_gt

        if self._trsf_extra['geom_augm'] is not None:
            geom_augm_det = self._trsf_extra['geom_augm'].to_deterministic()
            image = geom_augm_det.augment_image(image)
            image_gt = geom_augm_det.augment_image(image_gt)
            _, image_gt = cv2.threshold(image_gt, 127, 255, cv2.THRESH_BINARY)

        if self._trsf_extra['static_augm'] is not None:
            image = self._trsf_extra['static_augm'].augment_image(image)

        image, image_gt = Image.fromarray(image), Image.fromarray(image_gt)
        if self._to_tensor:
            tnsf = transforms.ToTensor()
            image, image_gt = tnsf(image), tnsf(image_gt)

        return image, image_gt
