import os
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from train import train_epoch
from validation import val_epoch
# from ResNet3D import resnet18
from improved_ResNet import resnet18
from Key_dataset import Key_Dataset

# Path setting
data_train_path = 'F:/dataset/CSL2018/gloss-zip/1/color_KCC_v2f_721/train/'
data_val_path = 'F:/dataset/CSL2018/gloss-zip/1/color_KCC_v2f_721/validate/'

label_path = "F:/dataset/CSL2018/gloss-zip/1/dictionary100.txt"
model_path = "F:/code/results/paper/model/improved_ResNet"
log_path = "F:/code/results/paper/log/improved_ResNet/improved_ResNet.log".format(datetime.now())
sum_path = "F:/code/runs/improved_ResNet"

# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 50
num_classes = 100
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-5
log_interval = 100
sample_size = 224
sample_duration = 16


# Train with 3DCNN
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # Load data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = Key_Dataset(data_path=data_train_path, label_path=label_path, frames=sample_duration,
                            num_classes=num_classes, train=True, transform=transform)
    val_set = Key_Dataset(data_path=data_val_path, label_path=label_path, frames=sample_duration,
                          num_classes=num_classes, val=True, transform=transform)
    logger.info("Dataset samples: {}".format(len(train_set) + len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Create model
    model = resnet18(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    exponent_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train_epoch(model, criterion, optimizer, exponent_schedule, train_loader, device, epoch, logger, log_interval, writer)
        # Validate the model
        val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
        # Save model
        slr_models = os.path.join(model_path, "resnet3d_epoch{:03d}.pth".format(epoch + 1))
        slr_models = slr_models.replace('\\', '/')
        torch.save(model.state_dict(), slr_models)
        logger.info("Epoch {} Model Saved".format(epoch + 1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
    writer.close()
