from numpy.random import randint

from torch.utils.data import Dataset

from synthetic_card_image_generator import SyntheticCardImageGenerator, generate_random_text, generate_random_face


class SyntheticCardImageDataset(Dataset):
    def __init__(self, image_size, transform=None):
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return 1000  # fake size of epoch

    def __getitem__(self, idx):
        card_image = SyntheticCardImageGenerator()

        for _ in range(25):
            text_image = generate_random_text(randint(5, 20))
            card_image.put_fragment(text_image)

        face_image = generate_random_face(randint(150, 350))

        card_image.put_fragment(face_image)

        card_image.draw_pattern_texts()

        return card_image.image
