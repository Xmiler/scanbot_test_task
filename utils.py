from datetime import datetime
from torch.nn import functional as F
import torch


def print_with_time(string):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time + ' - ' + string)


def get_latest_epoch_in_weights_folder(weights_path):
    filenames = list(weights_path.glob('*.pth'))
    if not filenames:
        return 0
    filename2epoch = lambda path: int(path.as_posix().split('_')[-1].split('.')[0])
    return max([filename2epoch(filename) for filename in filenames])


def pr_output_transform(output):
    y_pred, y = output
    y_pred = F.softmax(y_pred, dim=1)[:, 1:, :, :]
    y_pred = torch.round(y_pred)
    return y_pred, y
