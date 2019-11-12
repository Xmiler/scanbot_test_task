from numpy.random import randint
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from synthetic_card_image_generator import SyntheticCardImageGenerator, generate_random_text_image, generate_random_face_image


class SyntheticCardImageDataset(Dataset):
    SIZE = 256
    transform = transforms.Compose([transforms.ToTensor()])

    def __init__(self, to_tensor=False):
        self._to_tensor = to_tensor

    def __len__(self):
        return 1000  # fake size of epoch

    def __getitem__(self, idx):
        card_image = SyntheticCardImageGenerator()

        for _ in range(25):
            text_image = generate_random_text_image(randint(5, 20))
            card_image.put_fragment(text_image)

        face_image = generate_random_face_image(randint(150, 350))

        card_image.put_fragment(face_image)

        card_image.draw_pattern_texts()

        card_image.apply_square_and_zooming(factor=1.)

        card_image.resize(self.SIZE)

        image, image_gt = Image.fromarray(card_image.image), Image.fromarray(card_image.image_gt)

        if self._to_tensor:
            tnsf = transforms.ToTensor()
            image, image_gt = tnsf(image), tnsf(image_gt)

        return image, image_gt
