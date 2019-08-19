"""
第2章SSDで予測結果を画像として描画するクラス

"""
import numpy as np
import cv2  # OpenCVライブラリ
import torch

from ssd.model import DataTransform


class SSDPredict():

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # クラス名
        self.net = net  # SSDネットワーク

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする
        self.transform = DataTransform(input_size, color_mean)  # 前処理クラス


    def show(self, image, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image, data_confidence_level)

        self.vis_bbox_onlyperson(image, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image, data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        height, width, channels = image.shape  # 画像のサイズを取得

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            image, phase, "", "")  # アノテーションが存在しないので""にする。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # SSDで予測
        self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])

        detections = self.net(x)
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 条件以上の値を抽出
        # detections[:,0:,:,0] はクラスラベル0以外(背景以外)のすべてのbox(200)のconfの値
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)

#         print(find_index)        # find_indexはミニバッチ数、クラス、topのtuple
#         print(find_index[1][0])
#         print(find_index[1][1])
#         print(find_index[1][2])
        detections = detections[find_index]  #(number of detected Object, 5)

        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return  predict_bbox, pre_dict_label_index, scores

    def vis_bbox_onlyperson(self, image, bbox, label_index, scores, label_names):
        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        # BBox分のループ
        for i, bb in enumerate(bbox):
            # ラベル名
            label_name = label_names[label_index[i]]
            #追加
            if label_name != 'person':
                continue
            # 枠の座標
            x1,y1,x2,y2 = int(bb[0]), int(bb[1]),int(bb[2]),int(bb[3])
            # 長方形を描画する
            cv2.rectangle(image, (x1,y1), (x2,y2),(0,255,0),2)

        cv2.imshow("preview",image)
