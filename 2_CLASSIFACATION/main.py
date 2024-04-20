import os
import sys

if not os.path.exists("../datasets/15-Scene"):
    print("Please download the 15-scene dataset, https://figshare.com/articles/dataset/15-Scene_Image_Dataset/7007177")
    sys.exit(1)

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.cluster.vq import vq

from tqdm import tqdm

# 参数设置
TRAIN_RATIO = 150  # 每类图片用于训练的数量
NUM_CLUSTERS = 150  # K-means聚类中的类别数（视觉单词数量）
NUM_CLASSES = 15

# 路径设置
dataset_path = "../datasets/15-Scene"

# 初始化SIFT检测器
sift = cv2.SIFT()

# 用于保存特征和标签的列表
train_descriptors = []
train_labels = []
test_descriptors = []
test_labels = []

# 数据读取与特征提取
for category_id, category in enumerate(sorted(os.listdir(dataset_path))):
    folder_path = os.path.join(dataset_path, category)
    if os.path.isdir(folder_path):
        
        images = sorted(os.listdir(folder_path))
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path)
                keypoints, descriptors = sift.detectAndCompute(img, None)
                if descriptors is not None:
                    if i < TRAIN_RATIO:
                        train_descriptors.append(descriptors)
                        train_labels.append(category_id)
                    else:
                        test_descriptors.append(descriptors)
                        test_labels.append(category_id)
                        
print("Number of training images: ", len(train_descriptors))
# 将所有训练集描述符堆叠在一起进行K-means聚类
all_train_descriptors = np.vstack(train_descriptors)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
kmeans.fit(all_train_descriptors)


# 创建BoW特征表示
def extract_features(descriptors_list, kmeans):
    features = []
    for descriptors in descriptors_list:
        histogram = np.zeros(NUM_CLUSTERS)
        if descriptors is not None:
            words = vq(descriptors, kmeans.cluster_centers_)[0]
            for w in words:
                histogram[w] += 1
        features.append(histogram)
    return features


train_features = extract_features(train_descriptors, kmeans)
test_features = extract_features(test_descriptors, kmeans)

# 标准化特征
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# SVM分类器
svm = SVC(kernel='linear')
svm.fit(train_features, train_labels)
predicted_labels = svm.predict(test_features)

# 评估结果
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy:", accuracy)
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:\n", conf_matrix)
