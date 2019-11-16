from datetime import datetime


def print_with_time(string):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time + ' - ' + string)


def get_latest_epoch_in_weights_folder(weights_path):
    filenames = list(weights_path.glob('*.pth'))
    if not filenames:
        return 0
    filename2epoch = lambda path: int(path.as_posix().split('_')[-1].split('.')[0])
    return max([filename2epoch(filename) for filename in filenames])
