import io
import numpy as np
import PIL
from IPython.display import Image, display


def show_array(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    if a.ndim == 3:
        a = a[:, :, ::-1]  # BGR2RGB

    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
