from model.BCDNET import BCDNET
from utils.EvaluationNew import Evaluation, Index
import torchvision.transforms as Transforms
import time
import glob
import cv2
import numpy as np
import torch
from model.BFE_DPN  import BFExtractor
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print('Starting test...')
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define network
    BFENet = BFExtractor(n_channels=3, n_classes=1)
    BCDNet = BCDNET(n_channels=3, n_classes=1)

    # copy model to device
    BFENet.to(device=device)
    BCDNet.to(device=device)

    # Load model.pth
    BFENet.load_state_dict(torch.load('BestmIoU_BFE_DPN_epoch91_mIoU_91.864527.pth', map_location=device))
    BCDNet.load_state_dict(torch.load('BestmIoU_BCD_DPN_epoch91_mIoU_91.864527.pth', map_location=device))

    # Test status
    BFENet.eval()
    BCDNet.eval()

    trans = Transforms.Compose([Transforms.ToTensor()])

    # Load dataset path
    tests1_path = glob.glob('./samples/WHU/test/image1/*.tif')
    tests2_path = glob.glob('./samples/WHU/test/image2/*.tif')
    label_path = glob.glob('./samples/WHU/test/label/*.tif')

    # Define evaluation index
    IoU, c_IoU, uc_IoU, OA, Precision, Recall,   F1 = 0, 0, 0, 0, 0, 0, 0
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0, 0, 0, 0, 0, 0

    num=0

    f_acc = open('test_acc.txt', 'w')
    f_time = open('test_time.txt', 'w')
    with tqdm(total=len(label_path), desc='Test Epoch #{}'.format(num), ncols=160) as t:
        for tests1_path, tests2_path, label_path in zip(tests1_path, tests2_path, label_path):
            starttime = time.time()

            # Save path
            save_res_path = '.' + tests1_path.split('.')[1] + '_res.png'
            save_res_path = save_res_path.replace('image1', 'results')

            # Obtaining file name
            # Accoring to your own directory to modify the position of the split character.
            name = tests1_path.split('/')[5].split('.')[0]

            # Read images
            t1 = cv2.imread(tests1_path)
            t2 = cv2.imread(tests2_path)
            GT = cv2.imread(label_path)

            label_img = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)
            test1_img = trans(t1)
            test2_img = trans(t2)
            test1_img = test1_img.unsqueeze(0)
            test2_img = test2_img.unsqueeze(0)
            # Copy tensor to device
            test1_img = test1_img.to(device=device, dtype=torch.float32)
            test2_img = test2_img.to(device=device, dtype=torch.float32)

            # output prediction result
            list = []
            out1, feat1 = BFENet(test1_img)
            out2, feat2 = BFENet(test2_img)
            list.append(out1)
            list.append(out2)
            list.append(feat1)
            list.append(feat2)
            pred_Img = BCDNet(list)

            # Get prediction image
            pred = np.array(pred_Img.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0

            # print(num, tests1_path)
            # Save result
            cv2.imwrite(save_res_path, pred)
            #################### Show result ############################
            # io.imshow(pred)
            """
            pred_ = pred.astype(np.uint8)
            pred_ = np.expand_dims(pred_,2)
            pred_ = np.repeat(pred_, 3, axis=2)
            plt.figure(figsize=(10, 11))
            plt.suptitle('Building Change Detection Model')  # image
            plt.subplot(2, 2, 1), plt.title('T1-time image')
            plt.imshow(t1), plt.axis('off')
            plt.subplot(2, 2, 2), plt.title('T2-time image')
            plt.imshow(t2), plt.axis('off')
            plt.subplot(2, 2, 3), plt.title('Ground truth image')
            plt.imshow(GT), plt.axis('off')
            plt.subplot(2, 2, 4), plt.title('Change detection map')
            plt.imshow(pred_), plt.axis('off')
            plt.show()
            time.sleep(0.5)
            """
            ##############################################################
            endtime = time.time()
            if num == 0:
                f_time.write('each pair images time\n')
            f_time.write(str(num) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
                float('%2f' % (starttime - endtime))) + '\n')

            # Accuracy Evaluation
            monfusion_matrix = Evaluation(label=label_img, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or

            # Save val loss and accuracy
            if num == 1:
                f_acc.write('=================================================================================\n')
                f_acc.write('|Note: (num, FileName, TP, TN, FP, FN)|\n')
                f_acc.write('|Note: (ACC: FileName, OA, FA, MA, TE, mIoU, c_IoU, uc_IoU, Precision, Recall, F1)|\n')
                f_acc.write('=================================================================================\n')

            f_acc.write(str(num) + ',' + str(name) + '.tif' + ',' + str(TP) + ',' + str(TN) + ',' +
                        str(FP) + ',' + str(FN) + '\n')

            num += 1
            if num > 50:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
                FA, MA, TE = Indicators.CD_indicators()

                # print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=", str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
                #      str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=", str(float('%4f' % F1)))

            t.set_postfix({
                           'OA': OA,
                           'mIoU': '%.4f' % IoU,
                           'c_IoU': '%.4f' % c_IoU,
                           'uc_IoU': '%.4f' % uc_IoU,
                           'PRE': '%.4f' % Precision,
                           'REC': '%.4f' % Recall,
                           'F1': '%.4f' % F1})
            t.update(1)

    Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
    OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    FA, MA, TE = Indicators.CD_indicators()
    """
    print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=",
          str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
          str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=",
          str(float('%4f' % F1)))
    """
    f_acc.write('==========================================================================================================\n')
    f_acc.write('|SumConfusionMatrix:|  TP   |   TN   |  FP  |  FN   |\n')
    f_acc.write('|SumConfusionMatrix:|' + str(TPSum) + '|' + str(TNSum) + '|' + str(FPSum) + '|' + str(FNSum) + '|\n')
    f_acc.write('==========================================================================================================\n')
    f_acc.write('|TotalAcc:|   OA   |   FA   |   MA    |  TE   |  mIoU   |  c_IoU  | uc_IoU  |Precision| Recall  |   F1    |\n')
    f_acc.write('|TotalAcc:|' + str(float('%4f' % OA)) + '|' + str(float('%4f' % FA)) + '|' + str(float('%4f' % MA)) + '|' + str(float('%4f' % TE))
                + '|' + str(float('%4f' % IoU)) + '|' + str(float('%4f' % c_IoU)) + '|' + str(float('%4f' % uc_IoU)) + '|' +
                str(float('%4f' % Precision)) + '|' + str(float('%4f' % Recall)) + '|' + str(float('%4f' % F1)) + '|\n')
    f_acc.write(
        '==========================================================================================================\n')
    f_acc.close()
    f_time.close()