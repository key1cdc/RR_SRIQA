from torch.serialization import SourceChangeWarning
import warnings

warnings.filterwarnings("ignore", category=SourceChangeWarning)
import argparse
import torch
import htreg_data

torch.nn.Module.dump_patches = True
import torch.nn as nn
from torch.autograd import Variable
from htreg_data import *
import logging

parser = argparse.ArgumentParser(description="RRIQA")
parser.add_argument('--tdata_path', type=str, default="")
parser.add_argument('--sdata_path', type=str, default="")
parser.add_argument('--hreg_data', type=str, default="./train_data/train_data1.npy")
parser.add_argument('--model_path', type=str, default="./model/")
parser.add_argument("--nEpochs", type=int, default=20, help="number of epochs to train for")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate. Default=1e-4")
parser.add_argument("--resume", default='', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--checktime", default=1, type=int, help="step to confirm the feature validation")
parser.add_argument('--sr_method', type=str, default="method1", help="the selection of super resolution methods")
parser.add_argument("--logname", default='log', type=str, help="name of log file")


def htreg_test(model_path):
    test_path = "./data/test_data.npy"
    batch_size = 1
    dataset = htreg_data.HT_reg_test(test_path)
    dataloader = DataLoader(dataset, batch_size, drop_last=True)

    htreg_model = torch.load(model_path)["model"]
    htreg_model.eval()
    sbj_score = []
    pred_score = []
    pred_score1 = []

    for i, batch in enumerate(dataloader):
        with torch.no_grad():

            fg, s_score, label = batch

            fg = Variable(fg)
            label = Variable(label)
            label = torch.unsqueeze(label, dim=0)

            sbj_score.append(float(label))
            if torch.cuda.is_available():
                fg = fg.cuda()
                label = label.cuda()

            pred_s = htreg_model(fg)
            pred_s = pred_s.cpu().data[0].numpy().astype(np.float32)
            pred_score.append(round(float(pred_s), 6))
            pred_score1.append(float(pred_s))
            # print(i + 1)

    for n in range(len(sbj_score)):
        x = sbj_score[n]
        y = pred_score[n]
        print("sbj_score: {:.4f} pre_score: {:.4f}".format(x, y))


if __name__ == "__main__":
    model_path = "./model/htreg_model_epoch119.pth"
    htreg_test(model_path)
