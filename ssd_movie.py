import numpy as np
import cv2
import torch
from ssd.model import SSD
from ssd.predict import SSDPredict

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

ssd_config = {
    'num_classes': 21,  # 背景クラスを含めた合計クラス数
    'input_size': 300,  # 画像の入力サイズ
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

network = SSD(phase='inference',cfg=ssd_config)
network_weights = torch.load('weights/ssd300_mAP_77.43_v2.pth',
                        map_location={'cuda:0': 'cpu'})
network.load_state_dict(network_weights)

print('finished network loading')

cap = cv2.VideoCapture('movie/handsign.mp4')
ssd = SSDPredict(eval_categories=voc_classes, net=network)
while True:
    r,frame = cap.read()
    if r:
        img = cv2.resize(frame,(300,300))
        print(img.shape)
        ssd.show(img, data_confidence_level=0.6)

    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break
