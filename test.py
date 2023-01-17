import cv2
import os
import time
import math
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from scipy import signal
from sklearn import preprocessing
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from ResNet3D import resnet18
from improved_ResNet import resnet18
from OF import OpticalFlowCalculator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Key_Dataset(Dataset):
    def __init__(self, data_path, label_path, frames=16, num_classes=100, test=False, transform=None):
        super(Key_Dataset, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test = test
        self.flow = OpticalFlowCalculator()
        if self.test:
            self.videos_per_folder = int(250 * 0.1)
        self.data_folder = []
        try:
            obs_path = [os.path.join(self.data_path, item) for item in os.listdir(self.data_path)]
            self.data_folder = sorted([item for item in obs_path if os.path.isdir(item)])
        except Exception as e:
            print("Something wrong with your data path!!!")
            raise
        self.labels = {}
        try:
            label_file = open(self.label_path, 'r', encoding='UTF-8')
            for line in label_file.readlines():
                line = line.strip()
                line = line.split('\t')
                self.labels[line[0]] = line[1]
        except Exception as e:
            raise

    def read_images(self, folder_path):
        videoFile = cv2.VideoCapture(folder_path)
        ret, frame = videoFile.read()
        height = 0.2
        data = []
        count = 0
        imgs = []
        while ret:
            xvel, yvel = self.flow.processFrame(frame[120:720, 340:940])
            img = cv2.resize(frame[120:720, 340:940], (224, 224))
            imgs.append(img)
            if count >= 2:
                data.append(math.log(xvel * xvel + yvel * yvel + 1))
            ret, frame = videoFile.read()
            count += 1

        min_max_scaler = preprocessing.MinMaxScaler()
        data1 = np.squeeze(min_max_scaler.fit_transform(np.expand_dims(data, axis=1)), axis=1)
        peaks = signal.find_peaks(data1, height=height)
        begin = peaks[0][0] + 2
        end = peaks[0][-1] + 2
        body = end - begin + 1

        while body < self.frames:
            height = height / 2
            peaks = signal.find_peaks(data1, height=height)
            begin = peaks[0][0]+2
            end = peaks[0][-1]+2
            body = end - begin + 1
        assert body >= self.frames, "Too few images in your data folder: " + str(folder_path) + str(body)

        images = []
        K = []
        sta = begin - 1
        refer = imgs[sta]
        for j in range(0, self.frames):
            f = {}
            h = {}
            a = []
            start = int(sta + j * body / self.frames + 0.5)
            l = int(sta + (j + 1) * body / self.frames + 0.5)
            if l - start == 1:
                refer = imgs[l]
                K.append(l)
                image = self.transform(imgs[l])
                images.append(image)
            else:
                for m in range(start, l):
                    curr_frame = imgs[m]
                    diff = np.sqrt(np.sum(np.square(curr_frame - refer)))
                    f[m] = diff
                f_order = sorted(f.items(), key=lambda x: x[1], reverse=False)

                C = {}
                for n in range(len(f_order) - 1):
                    sum1 = []
                    sum2 = []
                    if n == 0:
                        sum1 = f_order[n][1]
                        m1 = f_order[n][1]
                        for q in range(1, len(f_order)):
                            single2 = f_order[q][1]
                            sum2.append(single2)
                        m2 = np.mean(sum2)
                        sigma1 = np.std(sum1, ddof=0)
                        sigma2 = np.std(sum2, ddof=0)
                    else:
                        for p in range(n + 1):
                            single1 = f_order[p][1]
                            sum1.append(single1)
                        m1 = np.mean(sum1)
                        for q in range(n + 1, len(f_order)):
                            single2 = f_order[q][1]
                            sum2.append(single2)
                        m2 = np.mean(sum2)
                        sigma1 = np.std(sum1, ddof=0)
                        sigma2 = np.std(sum2, ddof=0)
                    c = np.square(m1 - m2) / (sigma1 ** 2 + sigma2 ** 2)
                    C[f_order[n][0]] = c
                KK = max(C, key=lambda r: C[r])

                # 对后（n-KK）帧计算每一帧的模糊程度，选择模糊程度最低的一帧作为当前段的关键帧，也作为下一段的参考帧
                for y in range(len(f_order)):
                    Y = f_order[y][0]
                    a.append(Y)
                for b in a:
                    if b == KK:
                        index_KK = a.index(b)
                KK_latter = a[index_KK + 1:]
                # 利用拉普拉斯算子计算图像的模糊程度
                for e in KK_latter:
                    ima = imgs[e]
                    ima = cv2.Laplacian(ima, cv2.CV_64F).var()
                    h[e] = ima
                key = max(h, key=lambda r: h[r])
                image = self.transform(imgs[key])
                images.append(image)
                K.append(key)
                refer = imgs[key]

        images = torch.stack(images, dim=0)
        images = images.permute(1, 0, 2, 3)
        return images

    def __len__(self):
        return self.num_classes * self.videos_per_folder

    def __getitem__(self, idx):
        top_folder = self.data_folder[int(idx / self.videos_per_folder)]
        selected_folders = [os.path.join(top_folder, item) for item in os.listdir(top_folder)]
        selected_folders = sorted([item for item in selected_folders if os.path.isfile(item)])
        selected_folder = selected_folders[idx % self.videos_per_folder]
        images = self.read_images(selected_folder)
        label = torch.LongTensor([int(idx / self.videos_per_folder)])
        return {'data': images, 'label': label, 'images': images}

    def label_to_word(self, label):
        if isinstance(label, torch.Tensor):
            return self.labels['{:06d}'.format(label.item())]
        elif isinstance(label, int):
            return self.labels['{:06d}'.format(label)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='F:/dataset/CSL2018/gloss-zip/1/all_video_721/test',
                        type=str, help='Data path for testing')
    parser.add_argument('--label_path', default="F:/dataset/CSL2018/gloss-zip/1/dictionary100.txt",
                        type=str, help='Label path for testing')
    parser.add_argument('--model', default='resnet3d',
                        type=str, help='Choose a model for testing')
    parser.add_argument('--model_path', default='F:/code/results/paper/model/improved_ResNet/resnet3d_epoch038.pth',
                        type=str, help='Model state dict path')
    parser.add_argument('--batch_size', default=1,
                        type=int, help='Batch size for testing')
    parser.add_argument('--sample_size', default=224,
                        type=int, help='Sample size for testing')
    parser.add_argument('--sample_duration', default=16,
                        type=int, help='Sample duration for testing')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, dont use cuda')
    parser.add_argument('--cuda_devices', default='0',
                        type=str, help='Cuda visible devices')
    parser.add_argument('--num_classes', default=100,
                        type=int, help='Number of classes for testing')
    args = parser.parse_args()

    # Hyperparams
    num_classes = args.num_classes
    batch_size = args.batch_size
    sample_size = args.sample_size
    sample_duration = args.sample_duration
    data_path = args.data_path
    model_path = args.model_path
    label_path = args.label_path

    log_path = "C:/Users/218/Desktop/run/paper/improved_ResNet.log".format(datetime.now())
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger('SLR')

    # Use specific gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    # Device setting
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model
    if args.model =='resnet3d':
    model = resnet18(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes).to(device)

    # Run the model parallelly
    model = nn.DataParallel(model)
    # Load model
    model.load_state_dict(torch.load(model_path))
    # Test the model
    model.eval()
    all_pred = []
    count = 0
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    test_set = Key_Dataset(data_path=data_path, label_path=label_path, frames=sample_duration,
						   num_classes=num_classes, test=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    predictions = np.array([])
    labels = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # get the inputs and labels
            inputs = data['data'].to(device)
            # forward
            outputs = model(inputs)
            m = nn.Softmax(dim=1)
            outputs = m(outputs)
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1].cpu()
            predictions = np.append(predictions, int(prediction))
            labels = np.append(labels, int(data['label']))
            prob = torch.max(outputs, 1)[0].cpu().numpy()
            pro = prob[0]
            if data['label'] == prediction:
                count = count + 1

    acc = count / len(test_set)

    logger.info("model_path: {}".format(model_path))
    logger.info("all_count: {}".format(len(test_set)))
    logger.info("right_count: {}".format(count))
    logger.info("accuracy: {:.2f}%".format(acc * 100))
    logger.info("Running time: {:}s".format((end - start) / 2500))

    # 混淆矩阵
    C2 = confusion_matrix(labels, predictions, labels=np.arange(0, 100))
    print("classification report:\n{}".format(classification_report(labels, predictions)))

    accuracy = accuracy_score(labels, predictions, normalize=True, sample_weight=None)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)

    # 绘制混淆矩阵
    def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.viridis):
        """
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("显示百分比：")
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(cm)
        else:
            print('显示具体数字：')
            print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        plt.xlim(0, 99)
        plt.ylim(99, 0)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    plot_confusion_matrix(C2, normalize=True, title='Normalized confusion matrix')