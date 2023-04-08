import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.image as matimg
from torch.utils import data
import torchvision.transforms as transforms

from glob import glob
from eval import correlation
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from os import listdir
from os.path import join
from PIL import Image
import scipy.io as scio
import numpy as np



h_model = torch.load('./model/h_model_epoch140.pth')['model']
h_model.eval()
t_model = torch.load("./model/t1_model_epoch160.pth")["model"]
t_model.eval()

hreg_model = torch.load("./model/hreg_model_epoch181.pth")["model"]
hreg_model.eval()
treg_model = torch.load("./model/treg_model_epoch291.pth")["model"]
treg_model.eval()



class HT_reg_test(data.Dataset):

    def __init__(self, data_path):
        self.datas = np.load(data_path, allow_pickle=True).item()

        self.sr_s = self.datas["sr_s"]
        self.bls = self.datas["bls"]

        self.sr_y = self.datas["sr_y"]
        self.sr_t = self.datas["sr_t"]

        self.sr_s_mscn = self.datas["sr_s_mscn"]
        self.sr_t_mscn = self.datas["sr_t_mscn"]

        self.hw = self.datas["hw"]
        self.tw = self.datas["tw"]
        self.label = self.datas["label"]

        self.name = list(self.sr_y.keys())

    def __getitem__(self, item):
        data_name = self.name[item]

        s_map = loadmat("./data/qads_mat/" + data_name + ".mat", verify_compressed_data_integrity=False)["bls_map"]
        s_score = np.mean(s_map)
        s_score = s_score.astype(np.float32)

        sr_s = self.sr_s[data_name]
        bls = self.bls[data_name]

        sr_s_mscn = self.sr_s_mscn[data_name]
        sr_t_mscn = self.sr_t_mscn[data_name]

        sr_y = self.sr_y[data_name]
        sr_t = self.sr_t[data_name]

        ### crop weight
        hw = self.hw[data_name] * (350 * 470)
        tw = self.tw[data_name] * (350 * 470)
        label = self.label[data_name]

        sr_s = sr_s.astype(np.float32)
        bls = bls.astype(np.float32)
        sr_y = sr_y.astype(np.float32)
        sr_t = sr_t.astype(np.float32)

        sr_s_mscn = sr_s_mscn.astype(np.float32)
        sr_t_mscn = sr_t_mscn.astype(np.float32)

        hw = hw.astype(np.float32)
        tw = tw.astype(np.float32)

        # sr_s_crop = sr_s_mscn[15:365, 15:485]

        sr_s_crop = sr_s[15:365, 15:485] / 256
        sr_s_crop = transforms.ToTensor()(sr_s_crop)

        sr_t_crop = sr_t_mscn[15:365, 15:485]
        sr_t_crop = transforms.ToTensor()(sr_t_crop)

        ### HNet input
        sr_s = sr_s / 256
        sr_s1 = transforms.ToTensor()(sr_s)
        sr_s = torch.unsqueeze(sr_s1, dim=0)

        bls = bls / 256
        bls1 = transforms.ToTensor()(bls)
        bls = torch.unsqueeze(bls1, dim=0)

        ### TNet input
        sr_t_mscn1 = sr_t_mscn
        sr_t_mscn1 = transforms.ToTensor()(sr_t_mscn1)
        sr_t_mscn1 = torch.unsqueeze(sr_t_mscn1, dim=0)

        with torch.no_grad():
            if torch.cuda.is_available():
                sr_s = sr_s.cuda()
                bls = bls.cuda()

                sr_t_mscn1 = sr_t_mscn1.cuda()
            # print(sr_s.shape, bls.shape)
            h_feature, h_pred_map = h_model(sr_s, bls)
            h_feature = h_feature.cpu().data[0].numpy()
            h_pred_map = h_pred_map.cpu().data[0].numpy()

            t_feature, t_pred_map = t_model(sr_t_mscn1)
            t_feature = t_feature.cpu().data[0].numpy()
            t_pred_map = t_pred_map.cpu().data[0].numpy()

        hw = transforms.ToTensor()(hw)
        tw = transforms.ToTensor()(tw)

        tw1 = torch.unsqueeze(tw,dim=0)
        hw1 = torch.unsqueeze(hw,dim=0)

        sr_t_crop = torch.unsqueeze(sr_t_crop,dim=0)
        sr_s_crop = torch.unsqueeze(sr_s_crop, dim=0)

        h_feature1 = np.transpose(h_feature, (1, 2, 0))
        h_pred_map1 = np.transpose(h_pred_map, (1, 2, 0))
        h_feature1 = transforms.ToTensor()(h_feature1)
        h_pred_map1 = transforms.ToTensor()(h_pred_map1)
        h_feature1 = torch.unsqueeze(h_feature1, dim=0)
        h_pred_map1 = torch.unsqueeze(h_pred_map1, dim=0)

        t_feature1 = np.transpose(t_feature, (1, 2, 0))
        t_pred_map1 = np.transpose(t_pred_map, (1, 2, 0))
        t_feature1 = transforms.ToTensor()(t_feature1)
        t_pred_map1 = transforms.ToTensor()(t_pred_map1)
        t_feature1 = torch.unsqueeze(t_feature1, dim=0)
        t_pred_map1 = torch.unsqueeze(t_pred_map1, dim=0)



        with torch.no_grad():
            if torch.cuda.is_available():
                h_feature1 = h_feature1.cuda()
                h_pred_map1 = h_pred_map1.cuda()
                sr_s_crop = sr_s_crop.cuda()
                hw1 = hw1.cuda()

                t_feature1 = t_feature1.cuda()
                t_pred_map1 = t_pred_map1.cuda()
                sr_t_crop = sr_t_crop.cuda()
                tw1 = tw1.cuda()

            # print(t_feature1.shape, t_pred_map1.shape, sr_t_crop.shape, tw1.shape)

            tfc512, tpred_s = treg_model(t_feature1, t_pred_map1, sr_t_crop, tw1)
            tpred_s = tpred_s.cpu().data[0].numpy().astype(np.float32)

            hfc512, hpred_s = hreg_model(h_feature1, h_pred_map1, sr_s_crop, hw1)
            hpred_s = hpred_s.cpu().data[0].numpy().astype(np.float32)
            # pred_score.append(float(pred_s))

            fg = torch.cat([tfc512, hfc512], dim=-1)

        return fg, s_score, label

    def __len__(self):
        return len(self.name)

