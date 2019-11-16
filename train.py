from datetime import datetime
from pathlib import Path
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import SGD
from tensorboardX import SummaryWriter
from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

from dataset.synthetic_card_image_dataset import SyntheticCardImageDataset
from models.unet_mini import UNetMini
from utils import print_with_time, get_latest_epoch_in_weights_folder


print(' ================= Initialization ================= ')
EXPERIMENT_NAME = 'baseline2'
print(f'Experiment name: {EXPERIMENT_NAME}')

# --->>> Service parameters
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0xDEADFACE)
np.random.seed(0xDEADFACE)
torch.manual_seed(0xDEADFACE)

OUTPUT_PATH = Path('./artifacts/')
writer = SummaryWriter(OUTPUT_PATH / 'tensorboard' / EXPERIMENT_NAME)
WEIGHTS_PATH = OUTPUT_PATH / EXPERIMENT_NAME
DEVICE = "cuda"
CHECKPOINT_INTERVAL = 1
CHECKPOINT_TEMPLATE = "epoch_{}_{:d}.pth"


# --->>> Training parameters
BATCH_SIZE = 8
MAX_EPOCHS = 150
BASE_LR = 0.1
FAKE_EPOCH_SIZE = 1000

# model
model = UNetMini(2)
model.to(device=DEVICE)

optimizer = SGD(model.parameters(), lr=BASE_LR, momentum=0.9, weight_decay=5e-4)


def criterion(output, target):
    softmax = F.log_softmax(output, dim=1)
    return F.cross_entropy(softmax, torch.squeeze(target))


def adjust_learning_rate(optimizer, epoch):
    if epoch < 50:
        lr = BASE_LR
    else:
        lr = BASE_LR * (1 - (epoch-50) / (MAX_EPOCHS-50))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# data
dataset = SyntheticCardImageDataset(to_tensor=True, fake_epoche_size=FAKE_EPOCH_SIZE)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


# --->>> Callbacks
def update_lr_scheduler(engine):
    lr = adjust_learning_rate(optimizer, engine.state.epoch)
    print_with_time("Learning rate: {}".format(lr))
    writer.add_scalar('lr', lr, global_step=engine.state.epoch)


metrics = {'avg_loss': Loss(criterion)}
evaluator = create_supervised_evaluator(model, metrics=metrics, device=DEVICE)


def resume_latest_checkpoint(engine):
    epoch = get_latest_epoch_in_weights_folder(WEIGHTS_PATH)

    if epoch == 0:
        return

    state_dict = torch.load(WEIGHTS_PATH / f'epoch_model_{epoch}.pth')
    model.load_state_dict(state_dict)

    state_dict = torch.load(WEIGHTS_PATH / f'epoch_optimizer_{epoch}.pth')
    optimizer.load_state_dict(state_dict)

    engine.state.epoch = epoch
    engine.state.iteration = epoch * FAKE_EPOCH_SIZE

    print_with_time(f'Resumed to training state from epoch {epoch}.')


def compute_and_log_metrics(engine):
    epoch = engine.state.epoch
    metrics = evaluator.run(data_loader).metrics
    print_with_time("Validation Results - Epoch: {}  Average Loss: {:.4f}"
                    .format(engine.state.epoch, metrics['avg_loss']))
    writer.add_scalars('loss', {'validation': metrics['avg_loss']}, global_step=epoch)


def create_checkpoint(engine):
    epoch = engine.state.epoch

    if epoch % CHECKPOINT_INTERVAL != 0:
        return
    WEIGHTS_PATH.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), WEIGHTS_PATH / CHECKPOINT_TEMPLATE.format('model', epoch))
    torch.save(optimizer.state_dict(), WEIGHTS_PATH / CHECKPOINT_TEMPLATE.format('optimizer', epoch))
    print_with_time('Created checkpoint with training state.')


# evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# --->>> Trainer
trainer = create_supervised_trainer(model, optimizer, criterion, device=DEVICE)

# attach callbacks
trainer.add_event_handler(Events.STARTED, resume_latest_checkpoint)
trainer.add_event_handler(Events.STARTED, compute_and_log_metrics)

trainer.add_event_handler(Events.EPOCH_STARTED, update_lr_scheduler)

trainer.add_event_handler(Events.EPOCH_COMPLETED, compute_and_log_metrics)
trainer.add_event_handler(Events.EPOCH_COMPLETED, create_checkpoint)

print(' ================= Training! ================= ')
trainer.run(data_loader, max_epochs=MAX_EPOCHS)