from model.BCDNET import BCDNET
from model.BFE_DPN import BFExtractor
from utils.EvaluationNew import Evaluation, Index
from utils.dataset import Data_Loader
from torch import optim
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
# from tensorboardX import SummaryWriter

def train_net(net, BFENet, device, data_path, epochs=110, batch_size=4, lr=0.0001, ModelName='DPN_Inria', is_Transfer= True):

    if is_Transfer:
        print("Loading Transfer Learning Model.........")
        BFENet.load_state_dict(torch.load('Pretrain_BFE_'+ModelName+'_model_epoch75_mIoU_89.657089.pth', map_location=device))
    else:
        print("No Using Transfer Learning Model.........")

    # Load Dataset
    dataloader = Data_Loader(data_path=data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=dataloader,
                                   batch_size=batch_size,
                                   shuffle=True)

    # Define Optimizer
    optimizerBCDNet = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    optimizerBFENet = optim.Adam(BFENet.parameters(), lr=lr, weight_decay=1e-5)

    ##### This lr setting is used for LEVIR Dataset.
    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizerBCDNet, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)
    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizerBFENet, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)

    ##### This lr setting is used for WHUCD Dataset.
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizerBCDNet, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90], gamma=0.9)
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizerBFENet, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90], gamma=0.9)

    # 定义bec loss
    criterion = nn.BCEWithLogitsLoss()

    f_loss = open('train_loss.txt', 'w')
    f_time = open('train_time.txt', 'w')
    # 训练epochs次
    for epoch in range(1, epochs+1):
        BFENet.train()
        net.train()
        # train status
        # learning rate delay
        best_mIOU = float('0')

        num = int(0)
        starttime = time.time()

        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch), ncols=130, colour='white') as t:
            for image1, image2, label in train_loader:
                optimizerBCDNet.zero_grad()
                optimizerBFENet.zero_grad()

                # 将数据拷贝到device中
                image1 = image1.to(device=device)
                image2 = image2.to(device=device)
                label = label.to(device=device)

                # Output prediction result
                # image = torch.cat((image1, image2), 1)
                list = []  # 0: out1,1: out2,2: feat1,3: feat2
                out1, feat1 = BFENet(image1)
                out2, feat2 = BFENet(image2)
                list.append(out1)
                list.append(out2)
                list.append(feat1)
                list.append(feat2)
                pred = net(list)

                total_loss = criterion(pred, label)

                if num == 0:
                    if epoch == 0:
                        f_loss.write('Note: epoch (num, edge_loss, focal_loss, BCE_loss, total_loss)\n')
                        f_loss.write('epoch = ' + str(epoch) + '\n')
                    else:
                        f_loss.write('epoch = ' + str(epoch) + '\n')
                f_loss.write(str(num) + ',' + str(float('%5f' % total_loss)) + '\n')

                # Update
                total_loss.backward()
                optimizerBFENet.step()
                optimizerBCDNet.step()
                num += 1

                t.set_postfix({'lr': '%.5f' % optimizerBCDNet.param_groups[0]['lr'],
                                'loss': '%.4f' % (total_loss.item()),})
                t.update(1)
        # learning rate delay
        scheduler1.step()
        scheduler2.step()

        endtime = time.time()
        # val
        # if epoch > 10 and epoch % 2 == 0:
        # save model(pth)
        if epoch > 10:
            with torch.no_grad():
                mOA, IoU = val(BFENet, net, device, epoch)
                if best_mIOU < IoU:
                    best_mIOU = IoU
                    modelpath1 = 'BestmIoU_BFE_' + str(ModelName) + '_model_epoch' + str(epoch) + '_mIoU_' + str(float('%2f' % IoU)) + '.pth'
                    torch.save(BFENet.state_dict(), modelpath1)
                    modelpath2 = 'BestmIoU_BCD_' + str(ModelName) + '_model_epoch' + str(epoch) + '_mIoU_' + str(float('%2f' % IoU)) + '.pth'
                    torch.save(net.state_dict(), modelpath2)

        if epoch == 0:
            f_time.write('each epoch time\n')
        f_time.write(str(epoch)+','+str(starttime)+','+str(endtime)+','+str(float('%2f' % (starttime-endtime))) + '\n')

    f_loss.close()
    f_time.close()

def val(net1, net2, device, epoc):
    net1.eval()
    net2.eval()
    tests1_path = glob.glob('./samples/WHU/test/image1/*.tif')
    tests2_path = glob.glob('./samples/WHU/test/image2/*.tif')
    label_path = glob.glob('./samples/WHU/test/label/*.tif')
    trans = Transforms.Compose([Transforms.ToTensor()])
    TPSum = 0
    TNSum = 0
    FPSum = 0
    FNSum = 0
    C_Sum_or = 0
    UC_Sum_or = 0
    num = 0
    val_acc = open('val_acc.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    with tqdm(total=len(label_path), desc='Val Epoch #{}'.format(epoc), ncols=160, colour='yellow') as t:
        for tests1_path, tests2_path, label_path in zip(tests1_path, tests2_path, label_path):
            num += 1

            # Load image
            test1_img = cv2.imread(tests1_path)
            test2_img = cv2.imread(tests2_path)
            label_img = cv2.imread(label_path)
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
            test1_img = trans(test1_img)
            test2_img = trans(test2_img)
            test1_img = test1_img.unsqueeze(0)
            test2_img = test2_img.unsqueeze(0)
            test1_img = test1_img.to(device=device, dtype=torch.float32)
            test2_img = test2_img.to(device=device, dtype=torch.float32)

            # val reuslts
            list = []
            # image = torch.cat((test1_img, test2_img), 1)
            out1, feat1 = net1(test1_img)
            out2, feat2 = net1(test2_img)
            list.append(out1)
            list.append(out2)
            list.append(feat1)
            list.append(feat2)
            pred = net2(list)

            # Get prediction result
            pred = np.array(pred.data.cpu()[0])[0]

            # Evaluation
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            monfusion_matrix = Evaluation(label=label_img, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or

            if num > 30:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
                val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                              str(float('%2f' % (c_IoU))) + ',' +
                              'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                              'F1 = ' + str(float('%2f' % (F1))) + '\n')
            t.set_postfix({
                           'OA': '%.4f' % OA,
                           'mIoU': '%.4f' % IoU,
                           'c_IoU': '%.4f' % c_IoU,
                           'uc_IoU': '%.4f' % uc_IoU,
                           'PRE': '%.4f' % Precision,
                           'REC': '%.4f' % Recall,
                           'F1': '%.4f' % F1})
            t.update(1)

    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    return OA, IoU

if __name__ == '__main__':
    # Select device: If cuda else cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = BCDNET(n_channels=3, n_classes=1)
    BFENet = BFExtractor(n_channels=3, n_classes=1)

    BFENet.to(device=device)
    net.to(device=device)

    # Select dataset
    # data_path = "./samples/LEVIR/train"
    data_path = "./samples/WHU/train"
    train_net(net, BFENet, device, data_path)