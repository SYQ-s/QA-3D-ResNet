# 读视频进行均匀采样，生成帧数据集
import os
import cv2
import numpy as np
import math
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
from OF import OpticalFlowCalculator


train_all = 'F:/dataset/CSL2018/gloss-zip/1/video/train/'
val_all = 'F:/dataset/CSL2018/gloss-zip/1/video/validate/'
test_all = 'F:/dataset/CSL2018/gloss-zip/1/video/test/'

train_key = 'F:/dataset/CSL2018/gloss-zip/1/key/train/'
val_key = 'F:/dataset/CSL2018/gloss-zip/1/key/validate/'
test_key = 'F:/dataset/CSL2018/gloss-zip/1/key/test/'

train_files = os.listdir(train_all)
val_files = os.listdir(val_all)
test_files = os.listdir(test_all)
frames = 16
flow = OpticalFlowCalculator()
import warnings
warnings.filterwarnings("ignore")

def read_images(folder_path, key_path, frames):
    k_path, _ = key_path.split('.', 1)
    if not os.path.exists(k_path):
        os.mkdir(k_path)
    videoFile = cv2.VideoCapture(folder_path)
    ret, frame = videoFile.read()
    height = 0.2
    data = []
    count = 0
    imgs = []
    while ret:
        xvel, yvel = flow.processFrame(frame[120:720, 340:940])
        img = cv2.resize(frame[120:720, 340:940], (224, 224))
        imgs.append(img)
        if count >= 2:
            data.append(math.log(xvel * xvel + yvel * yvel + 1))
        ret, frame = videoFile.read()
        count += 1

    min_max_scaler = preprocessing.MinMaxScaler()
    data1 = np.squeeze(min_max_scaler.fit_transform(np.expand_dims(data, axis=1)), axis=1)
    peaks = signal.find_peaks(data1, height=height)
    begin = peaks[0][0]+2
    end = peaks[0][-1]+2

    body = end - begin + 1
    while body < frames:
        height = height/2
        peaks = signal.find_peaks(data1, height=height)
        begin = peaks[0][0]+2
        end = peaks[0][-1]+2
        body = end - begin + 1
    assert body >= frames, "Too few images in your data folder: " + str(folder_path) + str(body)

    images = []
    K = []
    sta = begin - 1
    refer = imgs[sta]
    for j in range(0, frames):
        f = {}
        h = {}
        a = []
        start = int(sta + j * body / frames + 0.5)
        l = int(sta + (j + 1) * body / frames + 0.5)

        if l - start == 1:
            refer = imgs[l]
            K.append(l)
            img = imgs[l]
            cv2.imwrite(k_path + '/' + "%d.jpg" % (l + 1), img)
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

            KK = max(C, key=lambda r: C[r])  # 25

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
            img = imgs[key]
            cv2.imwrite(k_path + '/' + "%d.jpg" % (key+1), img)
            K.append(key)
            refer = imgs[key]
    return images


for videos_file in train_files:
    if not os.path.exists(train_key + videos_file):
        os.mkdir(train_key + videos_file)
    videos_path = os.path.join(train_all, videos_file)
    train_videos = os.listdir(videos_path)
    for each_train_video in train_videos:
        path = train_all + videos_file + '/' + each_train_video
        key_path = train_key + videos_file + '/' + each_train_video
        indexes = read_images(path, key_path, frames)

for videos_file in val_files:
    if not os.path.exists(val_key + videos_file):
        os.mkdir(val_key + videos_file)
    videos_path = os.path.join(val_all, videos_file)
    val_videos = os.listdir(videos_path)
    for each_val_video in val_videos:
        path = val_all + videos_file + '/' + each_val_video
        key_path = val_key + videos_file + '/' + each_val_video
        indexes = read_images(path, key_path, frames)

for videos_file in test_files:
    if not os.path.exists(test_key + videos_file):
        os.mkdir(test_key + videos_file)
    videos_path = os.path.join(test_all, videos_file)
    test_videos = os.listdir(videos_path)
    for each_test_video in test_videos:
        path = test_all + videos_file + '/' + each_test_video
        key_path = test_key + videos_file + '/' + each_test_video
        indexes = read_images(path, key_path, frames)
