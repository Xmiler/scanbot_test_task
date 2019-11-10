import io
import string
from pathlib import Path

import numpy as np
from numpy.random import randint

import cv2
import PIL.Image

from trdg.computer_text_generator import generate
from trdg.utils import load_fonts

np.random.seed(0)

CHARACTERS = list(string.ascii_letters + string.digits * 3 + ':/' * 2)
FONTS = [font for i, font in enumerate(load_fonts('en')) if
         i in [1, 2, 15, 19, 22, 23, 24, 27, 35, 49]]  # picked MRZ like fonts
MAGES4AUGMENT_ROOT_PATH = Path(__file__).parent / '../images4augment/'

image_face_paths = sorted([path for path in (MAGES4AUGMENT_ROOT_PATH / 'faces').glob('**/*') if path.is_file()])


def create_image_with_text(text, font_id=None, font_size=None):
    if font_id is None:
        font_id = randint(0, high=len(FONTS))
    if font_size is None:
        font_size = randint(16, high=64)
    img = generate(text, font=FONTS[font_id], text_color='#000000', font_size=font_size, space_width=1., fit=False, orientation=0)
    return cv2.bitwise_not(np.array(img)[:, :, 3])  # library "trdg" writes result to alpha channel. Convert to grey.


def generate_random_text(text_len, font_id=None, font_size=None):
    text = ''.join(np.random.choice(CHARACTERS, size=text_len))
    return create_image_with_text(text, font_id=font_id, font_size=font_size)


def generate_random_face(size):
    image_face_path = np.random.choice(image_face_paths, 1)[0]
    image_face = cv2.imread(image_face_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    h_, w_ = image_face.shape
    size_ = max(h_, w_)
    h = int(h_ * size / size_)
    w = int(w_ * size / size_)
    image_face = cv2.resize(image_face, (w, h))
    return image_face


class SyntheticCardImageGenerator:
    CARD_SIZE = (650, 800)
    PATTERN_TEXTS_NUM = 3

    def __init__(self):

        self._image = 255 * np.ones(self.CARD_SIZE, dtype=np.uint8)

        # generate bare pattern texts
        self._pattern_texts = [self.generate_pattern_text_bare() for _ in range(self.PATTERN_TEXTS_NUM)]

        # set patternt texts drawing parameters
        pt_xs = randint(0, high=100) * np.ones((self.PATTERN_TEXTS_NUM), dtype=np.int)
        pt_ys = self.CARD_SIZE[0] - np.cumsum(randint(50, high=100, size=(self.PATTERN_TEXTS_NUM)))
        pt_ys -= 10
        self._pattern_text_imgs_pt = list(zip(pt_ys, pt_xs))
        font_id = randint(0, high=len(FONTS))
        font_size = 60

        # draw bare pattern texts
        self._pattern_text_imgs = [create_image_with_text(pattern_text, font_id=font_id, font_size=font_size) for
                                   pattern_text in self._pattern_texts]
        self.draw_pattern_texts()

        # store ground truth
        self._image_gt = cv2.bitwise_not(self._image)

        # redraw with non-bare pattern text
        self._pattern_texts = [
            ''.join([ch if ch == '<' else np.random.choice(list(string.ascii_uppercase), 1)[0] for ch in pattern_text]) for
            pattern_text in self._pattern_texts]
        self._pattern_text_imgs = [create_image_with_text(pattern_text, font_id=font_id, font_size=font_size) for
                                   pattern_text in self._pattern_texts]
        self.draw_pattern_texts()

    @property
    def image(self):
        return self._image

    @property
    def image_gt(self):
        return self._image_gt

    def put_fragment(self, fragment, pt=None):
        if pt is None:
            pt = tuple([randint(0, high=self.CARD_SIZE[i]) for i in [0, 1]])
        size = (min(self._image.shape[0] - pt[0], fragment.shape[0]),
                min(self._image.shape[1] - pt[1], fragment.shape[1]))
        self._image[pt[0]:pt[0] + size[0], pt[1]:pt[1] + size[1]] = fragment[:size[0], :size[1]]

    def draw_pattern_texts(self):
        for img_i in range(self.PATTERN_TEXTS_NUM):
            self.put_fragment(self._pattern_text_imgs[img_i], self._pattern_text_imgs_pt[img_i])

    def apply_square_and_zooming(self, factor=1.):
        size = int(factor*max(self.CARD_SIZE))

        new_image = np.zeros([size] * 2, dtype=np.uint8)
        new_image_gt = np.zeros([size] * 2, dtype=np.uint8)

        y_delta, x_delta = ((size - np.array(self.CARD_SIZE)) / 2).astype(np.int)

        for obj_dst, obj_src in [(new_image, self._image), (new_image_gt, self._image_gt)]:
            obj_dst[y_delta:self.CARD_SIZE[0] + y_delta, x_delta:self.CARD_SIZE[1] + x_delta] = obj_src

        self._image = new_image
        self._image_gt = new_image_gt

    @staticmethod
    def generate_pattern_text_bare(max_size=16, pieces_num=10, piece_max_len=20):
        while True:
            rnd_flags = randint(0, high=2, size=pieces_num)
            rnd_lens = randint(1, high=piece_max_len + 1, size=(pieces_num))

            pieces = [(' ' if f else '<') * n for f, n in zip(rnd_flags, rnd_lens)]
            pattern_text_bare = ''.join(pieces)[:max_size]

            if '<' in pattern_text_bare:
                break
        return pattern_text_bare
