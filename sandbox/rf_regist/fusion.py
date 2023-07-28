from __future__ import print_function
import numpy as np
from feature.affine_ransac import Ransac
from feature.align_transform import Align
from feature.affine_transform import Affine
import argparse
import os
from rgb_mask import rgb_mask
from scipy.ndimage import shift
from skimage import registration
from pystackreg import StackReg
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model
from keras.models import Sequential
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from glob import glob
import scipy
import keras
import MBLLEN.main.Network as Network
import MBLLEN.main.utls as utls
import sys
import time
from PIL import Image, ImageEnhance

sys.path.append("rf/quick_start/")
from coarseAlignFeatMatch import CoarseAlign


sys.path.append("rf/utils/")
import outil

sys.path.append("rf/model/")
import model as model

import PIL.Image as Image
import os
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle
import pandas as pd
import kornia.geometry as tgm
import cv2
from itertools import product

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import matplotlib.pyplot as plt


def ssim_loss(img1, img2):
    size = 11
    sigma = 1.5
    # SSIM
    # ssim_loss_value=tf.image.ssim(img1,img2,max_val=1,filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)
    ssim_loss_value = tf.image.ssim(
        tf.convert_to_tensor(img1),
        tf.convert_to_tensor(img2),
        max_val=255,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
    )
    # Lp

    pixel_loss = tf.sqrt(tf.reduce_sum(tf.square(img1 - img2)))
    ssim_loss = 1 - ssim_loss_value
    loss = 10 * ssim_loss + pixel_loss
    return loss


# ENCODER
# creating the First Layer
def recon_model(input_shape):
    inimg = Input(shape=input_shape)
    conv1 = Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="SAME")(inimg)

    # creating the Dense Block
    DC1 = Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="SAME")(conv1)

    DC2_input = Concatenate(axis=3)([conv1, DC1])
    DC2 = Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="SAME")(DC2_input)

    DC3_input = Concatenate(axis=3)([conv1, DC1, DC2])
    DC3 = Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="SAME")(DC3_input)
    DC4_input = Concatenate(axis=3)([conv1, DC1, DC2, DC3])

    # Decoder
    Conv2 = Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="SAME")(DC4_input)
    Conv3 = Conv2D(32, kernel_size=3, strides=1, activation="relu", padding="SAME")(Conv2)
    Conv4 = Conv2D(16, kernel_size=3, strides=1, activation="relu", padding="SAME")(Conv3)
    Conv5 = Conv2D(1, kernel_size=3, strides=1, activation="relu", padding="SAME")(Conv4)

    Autoencoder = Model(inputs=inimg, outputs=Conv5)
    Autoencoder.compile(optimizer="adam", loss=ssim_loss, metrics=["accuracy"])

    return Autoencoder


def registeration(mode, source, moving, contrast=0.04, edge=10, ocatave=3):
    af = Affine()

    outlier_rate = 0.9
    A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate)

    K = 3
    idx = np.random.randint(0, pts_s.shape[1], (K, 1))
    A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

    rs = Ransac(K=3, threshold=1)

    residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

    A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)

    al = Align(
        source, moving, mode=mode, threshold=100, contrast=contrast, edge=edge, octave=ocatave
    )

    registered = al.align_image()

    return registered


def mode_reg(l, img_path, mode, nm):
    reg_list = []
    if mode == "avg":
        for item in l:
            img_read = cv2.imread(img_path + "/" + item)
            reg_list.append(img_read)
    else:
        target = cv2.imread(img_path + "/" + l[0])
        reg_list.append(target)
        for i in range(len(l) - 1):
            _, reg = registeration(mode, img_path + "/" + l[i + 1], img_path + "/" + l[0])
            # cv2.imwrite(img_path + "/" + mode + "/" + str(nm) + str(i) + "_" + str(len(l)) + ".png", reg)
            reg_list.append(reg)

    # mean
    myList = [value for value in reg_list if value.any() != None]
    n = len(myList)
    sum = 0
    for i in range(n):
        sum += myList[i] * (1 / n)

    registered = np.uint8(sum)

    cv2.imwrite(img_path + "/" + mode + "/" + str(nm) + ".png", registered)


def enhancement(model, path_all, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(len(path_all)):
        img_A_path = path_all[i]
        img_A = utls.imread_color(img_A_path)

        img_A = img_A[np.newaxis, :]

        starttime = time.perf_counter()
        out_pred = model.predict(img_A)
        endtime = time.perf_counter()
        print("The " + str(i + 1) + "th image's Time:" + str(endtime - starttime) + "s.")
        fake_B = out_pred[0, :, :, :3]

        fake_B_o = fake_B

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        percent_max = sum(sum(gray_fake_B >= maxrange)) / sum(sum(gray_fake_B <= 1.0))
        # print(percent_max)
        max_value = np.percentile(gray_fake_B[:], highpercent)
        if percent_max < (100 - highpercent) / 100.0:
            scale = maxrange / max_value
            fake_B = fake_B * scale
            fake_B = np.minimum(fake_B, 1.0)

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        sub_value = np.percentile(gray_fake_B[:], lowpercent)
        fake_B = (fake_B - sub_value) * (1.0 / (1 - sub_value))

        imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(imgHSV)
        S = np.power(S, hsvgamma)
        imgHSV = cv2.merge([H, S, V])
        fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
        fake_B = np.minimum(fake_B, 1.0)

        if flag:
            outputs = np.concatenate([img_A[0, :, :, :], fake_B_o, fake_B], axis=1)
        else:
            outputs = fake_B

        filename = os.path.basename(path_all[i])
        img_name = output_path + "/" + filename
        # scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(img_name)
        outputs = np.minimum(outputs, 1.0)
        outputs = np.maximum(outputs, 0.0)
        utls.imwrite(img_name, fake_B)


def unblur_image(image, sharpness_level):
    # Create a kernel for sharpening
    kernel = np.array([[-1, -1, -1], [-1, sharpness_level, -1], [-1, -1, -1]])

    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)

    # Display the original and sharpened images

    return sharpened_image


import cv2
import numpy as np


def auto_adjust_images(image1, image2):
    # Convert images to HSV color space
    image1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    image2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Extract intensity channel
    image1_intensity = image1_hsv[:, :, 2].astype(np.float32)
    image2_intensity = image2_hsv[:, :, 2].astype(np.float32)

    # Calculate mean intensity
    mean_intensity1 = np.mean(image1_intensity)
    mean_intensity2 = np.mean(image2_intensity)

    # Calculate scaling factor
    scaling_factor = mean_intensity2 / mean_intensity1

    # Adjust intensity channel of image1
    adjusted_intensity1 = image1_intensity * scaling_factor
    adjusted_intensity1 = np.clip(adjusted_intensity1, 0, 255).astype(np.uint8)

    # Merge adjusted intensity channel with original color channels of image1
    image1_hsv[:, :, 2] = adjusted_intensity1

    # Convert image1 back to BGR color space
    adjusted_image1 = cv2.cvtColor(image1_hsv, cv2.COLOR_HSV2BGR)

    return adjusted_image1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ref", help="ref image name")
    # parser.add_argument("--moving", help="moving image name")
    parser.add_argument("-p", "--path", help="img path")
    parser.add_argument("-rpm", "--repeat_mode", help="orb, sift, akaze, cnn, avg", default="sift")
    parser.add_argument("-bm", "--blue_mode", help="orb, sift, akaze, cnn, avg", default="sift")
    parser.add_argument("-gm", "--green_mode", help="orb, sift, akaze, cnn, avg", default="sift")
    parser.add_argument("-rm", "--red_mode", help="orb, sift, akaze, cnn, avg", default="sift")

    args = parser.parse_args()
    # source_path = args.ref
    # target_path = args.moving
    img_path = args.path
    rpm = args.repeat_mode
    bm = args.blue_mode
    gm = args.green_mode
    rm = args.red_mode
    run_num = img_path[12:]
    save_path = img_path[:11] + "/result"
    print("Run:             ", run_num)
    print(save_path)

    # print("当前路径: ", os.getcwd())
    for filename in os.listdir(img_path):
        newName = str(filename)
        newName = newName.replace(" ", "_")
        os.rename(os.path.join(img_path, filename), os.path.join(img_path, newName))
    img_list = os.listdir(img_path)
    if not os.path.exists(img_path + "/" + rpm):
        os.mkdir(img_path + "/" + rpm)

    # save path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path + "/result_" + run_num[3:]):
        os.mkdir(save_path + "/result_" + run_num[3:])
    if not os.path.exists(save_path + "/result_" + run_num[3:] + "/" + run_num + "_reg"):
        os.mkdir(save_path + "/result_" + run_num[3:] + "/" + run_num + "_reg")
    if not os.path.exists(save_path + "/result_" + run_num[3:] + "/" + run_num + "_merg"):
        os.mkdir(save_path + "/result_" + run_num[3:] + "/" + run_num + "_merg")
    if not os.path.exists(save_path + "/result_" + run_num[3:] + "/" + run_num + "_white"):
        os.mkdir(save_path + "/result_" + run_num[3:] + "/" + run_num + "_white")
    if not os.path.exists(save_path + "/result_" + run_num[3:] + "/enhance_" + run_num):
        os.mkdir(save_path + "/result_" + run_num[3:] + "/enhance_" + run_num)
    if not os.path.exists(save_path + "/result_" + run_num[3:] + "/" + run_num + "_dark"):
        os.mkdir(save_path + "/result_" + run_num[3:] + "/" + run_num + "_dark")

    # repeated
    # img440 = [s for s in img_list if '440' in s]
    """img450 = [s for s in img_list if '450' in s]
    img460 = [s for s in img_list if '460' in s]
    img470 = [s for s in img_list if '470' in s]
    img480 = [s for s in img_list if '480' in s]
    img490 = [s for s in img_list if '490' in s]
    img500 = [s for s in img_list if '500' in s]
    img510 = [s for s in img_list if '510' in s]
    img520 = [s for s in img_list if '520' in s]
    img530 = [s for s in img_list if '530' in s]
    img540 = [s for s in img_list if '540' in s]
    img550 = [s for s in img_list if '550' in s]
    img560 = [s for s in img_list if '560' in s]
    img570 = [s for s in img_list if '570' in s]
    img580 = [s for s in img_list if '580' in s]
    img590 = [s for s in img_list if '590' in s]
    img600 = [s for s in img_list if '600' in s]
    img610 = [s for s in img_list if '610' in s]
    img620 = [s for s in img_list if '620' in s]
    img630 = [s for s in img_list if '630' in s]
    img640 = [s for s in img_list if '640' in s]
    img650 = [s for s in img_list if '650' in s]
    img660 = [s for s in img_list if '660' in s]
    img670 = [s for s in img_list if '670' in s]
    img680 = [s for s in img_list if '680' in s]
    img690 = [s for s in img_list if '690' in s]
    img700 = [s for s in img_list if '700' in s]
    img710 = [s for s in img_list if '710' in s]
    img720 = [s for s in img_list if '720' in s]

    #mode_reg(img440, img_path, rpm, 440)
    #mode_reg(img450, img_path, rpm, 450)
    mode_reg(img460, img_path, rpm, 460)
    mode_reg(img470, img_path, rpm, 470)
    mode_reg(img480, img_path, rpm, 480)
    mode_reg(img490, img_path, rpm, 490)
    mode_reg(img500, img_path, rpm, 500)
    mode_reg(img510, img_path, rpm, 510)
    mode_reg(img520, img_path, rpm, 520)
    mode_reg(img530, img_path, rpm, 530)
    mode_reg(img540, img_path, rpm, 540)
    mode_reg(img550, img_path, rpm, 550)
    mode_reg(img560, img_path, rpm, 560)
    mode_reg(img570, img_path, rpm, 570)
    mode_reg(img580, img_path, rpm, 580)
    mode_reg(img590, img_path, rpm, 590)
    mode_reg(img600, img_path, rpm, 600)
    mode_reg(img610, img_path, rpm, 610)
    mode_reg(img620, img_path, rpm, 620)
    mode_reg(img630, img_path, rpm, 630)
    mode_reg(img640, img_path, rpm, 640)
    mode_reg(img650, img_path, rpm, 650)
    mode_reg(img660, img_path, rpm, 660)
    mode_reg(img670, img_path, rpm, 670)
    mode_reg(img680, img_path, rpm, 680)
    mode_reg(img690, img_path, rpm, 690)
    mode_reg(img700, img_path, rpm, 700)
    mode_reg(img710, img_path, rpm, 710)
    mode_reg(img720, img_path, rpm, 720)"""

    # enhancement
    flag = 1
    lowpercent = 5
    highpercent = 95
    maxrange = 8 / 10.0
    hsvgamma = 8 / 10.0

    """input_folder_all = img_path + "/" + rpm
    path_all = glob(input_folder_all + '/*.*')
    model_name = 'Syn_img_lowlight_withnoise'
    mbllen = Network.build_mbllen((None, None, 3))
    mbllen.load_weights('MBLLEN/models/' + model_name + '.h5')
    opt = tf.keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mbllen.compile(loss='mse', optimizer=opt)

    enhancement(mbllen, path_all, img_path + "/enhance")"""

    # fusion
    """Autoencoder = recon_model(input_shape=(None, None, 1))
    Autoencoder.load_weights('loop/model/HS622.h5')
    Autoencoder.compile(optimizer='adam', loss=ssim_loss, metrics=['accuracy'])

    imgI = cv2.imread(img_path + '/' + rpm + '/630.png', 0)

    w, h = imgI.shape[0], imgI.shape[1]

    encoder = Model(inputs=Autoencoder.layers[0].input, outputs=Autoencoder.layers[7].output)

    decoder = Sequential()
    for layer in Autoencoder.layers[8:]:
        decoder.add(layer)

    b_list = []
    g_list = []
    r_list = []

    #path = save_path+"/result_"+run_num[3:]+"/enhance_"+run_num
    path = img_path + "/enhance"

    for i in range(5):
        c = cv2.imread(path + '/' + str(460 + 10 * i) + '.png', 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        b_list.append(c)

    for i in range(10):
        c = cv2.imread(path + '/' + str(510 + 10 * i) + '.png', 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        g_list.append(c)

    for i in range(10):

        c = cv2.imread(path + '/' + str(610 + 10 * i) + '.png', 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        r_list.append(c)

    imgEB_list = []
    for img in b_list:
        img = np.array(img).reshape(-1, w, h, 1)
        imgE = encoder.predict(img)
        imgEB_list.append(imgE)

    imgEG_list = []
    for img in g_list:
        img = np.array(img).reshape(-1, w, h, 1)
        imgE = encoder.predict(img)
        imgEG_list.append(imgE)

    imgER_list = []
    for img in r_list:
        img = np.array(img).reshape(-1, w, h, 1)
        imgE = encoder.predict(img)
        imgER_list.append(imgE)

    fuse_b = sum(imgEB_list) / len(imgEB_list)
    fuse_B = decoder.predict(fuse_b)

    fuse_g = sum(imgEG_list) / len(imgEG_list)
    fuse_G = decoder.predict(fuse_g)

    fuse_r = sum(imgER_list) / len(imgER_list)
    fuse_R = decoder.predict(fuse_r)"""
    b_list = []
    g_list = []
    r_list = []

    # path = save_path+"/result_"+run_num[3:]+"/enhance_"+run_num
    path = img_path + "/enhance"

    for i in range(5):
        c = cv2.imread(path + "/" + str(460 + 10 * i) + ".png", 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        b_list.append(c)

    for i in range(10):
        c = cv2.imread(path + "/" + str(510 + 10 * i) + ".png", 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        g_list.append(c)

    for i in range(10):
        c = cv2.imread(path + "/" + str(610 + 10 * i) + ".png", 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        r_list.append(c)

    fuse_B = sum(b_list) / len(b_list)

    fuse_G = sum(g_list) / len(g_list)

    fuse_R = sum(r_list) / len(r_list)

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue.png", fuse_B)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/green.png", fuse_G)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/red.png", fuse_R)

    # sharp
    """b = cv2.imread(save_path+'/result_'+run_num[3:]+'/blue.png', 0)
    g = cv2.imread(save_path+'/result_'+run_num[3:]+'/green.png', 0)
    r = cv2.imread(save_path+'/result_'+run_num[3:]+'/red.png', 0)

    ib = cv2.imread(save_path+'/result_'+run_num[3:]+'/blue.png', flags=cv2.IMREAD_COLOR)
    ig = cv2.imread(save_path+'/result_'+run_num[3:]+'/green.png', flags=cv2.IMREAD_COLOR)
    ir = cv2.imread(save_path+'/result_'+run_num[3:]+'/red.png', flags=cv2.IMREAD_COLOR)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    ib_sharp = cv2.filter2D(src=ib, ddepth=-1, kernel=kernel)
    ig_sharp = cv2.filter2D(src=ig, ddepth=-1, kernel=kernel)
    ir_sharp = cv2.filter2D(src=ir, ddepth=-1, kernel=kernel)

    cv2.imwrite(save_path+'/result_'+run_num[3:]+'/blue.png', ib_sharp)
    cv2.imwrite(save_path+'/result_'+run_num[3:]+'/green.png', ig_sharp)
    cv2.imwrite(save_path+'/result_'+run_num[3:]+'/red.png', ir_sharp)"""

    """factor = 25
    imb = Image.open(save_path+'/result_'+run_num[3:]+'/blue.png')
    enhancer = ImageEnhance.Sharpness(imb)
    im_s_1 = enhancer.enhance(factor)
    im_s_1.save(save_path+'/result_'+run_num[3:]+'/blue.png');

    img = Image.open(save_path+'/result_'+run_num[3:]+'/green.png')
    enhancer = ImageEnhance.Sharpness(img)
    im_s_1 = enhancer.enhance(factor)
    im_s_1.save(save_path+'/result_'+run_num[3:]+'/green.png');

    imr = Image.open(save_path+'/result_'+run_num[3:]+'/red.png')
    enhancer = ImageEnhance.Sharpness(imr)
    im_s_1 = enhancer.enhance(factor)
    im_s_1.save(save_path+'/result_'+run_num[3:]+'/red.png');"""

    """imb = cv2.imread(save_path+'/result_'+run_num[3:]+'/blue.png')
    alpha = 2  # Contrast control
    beta = 15  # Brightness control
    # call convertScaleAbs function
    sharpness_level = 9
    # Call the function to unblur/sharpen the image with the specified sharpness level
    adjustb = unblur_image(imb, sharpness_level)
    adjustb = cv2.convertScaleAbs(adjustb, alpha=alpha, beta=beta)

    cv2.imwrite(save_path+'/result_'+run_num[3:]+'/blue.png',adjustb)

    img = cv2.imread(save_path + '/result_' + run_num[3:] + '/green.png')
    alpha = 1.5  # Contrast control
    beta = 12  # Brightness control
    # call convertScaleAbs function
    sharpness_level = 9
    # Call the function to unblur/sharpen the image with the specified sharpness level
    adjustg = unblur_image(img, sharpness_level)
    adjustg = cv2.convertScaleAbs(adjustg, alpha=alpha, beta=beta)

    cv2.imwrite(save_path + '/result_' + run_num[3:] + '/green.png', adjustg)"""

    imr = cv2.imread(save_path + "/result_" + run_num[3:] + "/red.png", 0)
    imb = cv2.imread(save_path + "/result_" + run_num[3:] + "/blue.png", 0)
    img = cv2.imread(save_path + "/result_" + run_num[3:] + "/green.png", 0)
    # alpha = 1.5  # Contrast control
    # beta = 0  # Brightness control
    # call convertScaleAbs function
    # adjustr = cv2.convertScaleAbs(imr, alpha=alpha, beta=beta)
    # cv2.imwrite(save_path + '/result_' + run_num[3:] + '/red.png', adjustr)
    im_merg = cv2.merge((imb, img, imr))

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/im_merg.png", im_merg)

    # registration b, g, r
    # built target

    """if run_num == "Run05" or run_num == "Run06" or run_num == "Run07":
        target = cv2.imread('brain/HS001/Labelling data/Definitive segmentation/HS001 - Trial with capillaries excluded/'
                            'Run 04-08/Patient_1_2021-03-28_10-26-19_I.JPG')
    if run_num == "Run04":
        target = cv2.imread('brain/HS001/Labelling data/Definitive segmentation/HS001 - Trial with capillaries excluded/'
                            'Run 11-12/Patient_1_2021-03-28_10-23-51_I.JPG')

    if run_num == "Run18":
        target = cv2.imread('brain/HS001/Labelling data/Definitive segmentation/HS001 - Trial with capillaries excluded/'
                            'Run 13-18/Patient_1_2021-03-28_11-23-08_I.JPG')
    if run_num == "Run19" or run_num == "Run02":
        target = cv2.imread('brain/HS001/Labelling data/Definitive segmentation/HS001 - Trial with capillaries excluded/'
                            'Run 19-24/Patient_1_2021-03-28_12-44-20_I.JPG')

    if run_num == "Run08":
        target = cv2.imread('brain/HS020/Labelling data/Definitive segmentations/HS020 Capillary excluded/'
                            'Run 08 and 10/Patient_26  16-26-19.JPG')"""
    # if run_num == "Run08" or run_num == "Run10":
    # target = cv2.imread(
    #'brain/HS004/Labelling data/Definitive segmentations/HS004 - capillary excluded/Run 07-10/'
    #'Patient_1_2021-05-16_11-28-01_I.JPG')
    """if run_num == "Run05":
        target = cv2.imread(
            'brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 05-06/'
            'Patient_2_2021-07-19_11-27-51_I.JPG')

    if run_num == "Run07":
        target = cv2.imread(
            'brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 07/'
            'Patient_2_2021-07-19_11-31-22_I.JPG')
    if run_num == "Run08":
        target = cv2.imread(
            'brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 08-09/'
            'Patient_2_2021-07-19_12-03-01_I.JPG')

    if run_num == "Run11":
        target = cv2.imread('brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 10-11/'
                            'Patient_2_2021-07-19_12-04-23_I.JPG')
    if run_num == "Run12":
        target = cv2.imread('brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 12/'
                            'Patient_2_2021-07-19_12-05-49_I.JPG')
    if run_num == "Run13":
        target = cv2.imread('brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 13/'
                            'Patient_2_2021-07-19_14-22-02_I.JPG')
    if run_num == "Run14":
        target = cv2.imread('brain/HS005/Labelling data/Definitive segmentations/HS005 - capillary excluded/Run 14/'
                            'Patient_2_2021-07-19_14-22-05_I.JPG')"""

    # if run_num == "Run04" or run_num == "Run06":
    # target = cv2.imread(
    #'brain/HS022/Labelling data/Definitive segmentations/HS022 Capillaries excluded/Run04-05/'
    #'__2022-05-26_10-43-54_I.JPG')
    """if run_num == "Run04":
        target = cv2.imread(
            'brain/HS020/Labelling data/Definitive segmentations/HS020 Capillary excluded/Run 04-05/'
            'Patient_26  14-39-35.JPG')
    if run_num == "Run06":
        target = cv2.imread(
            'brain/HS020/Labelling data/Definitive segmentations/HS020 Capillary excluded/Run 06-07/'
            'Patient_26  15-04-50.JPG')
    if run_num == "Run08":
        target = cv2.imread(
            'brain/HS020/Labelling data/Definitive segmentations/HS020 Capillary excluded/Run 08 and 10/'
            'Patient_26  16-26-19.JPG')
    if run_num == "Run09":
        target = cv2.imread(
            'brain/HS020/Labelling data/Definitive segmentations/HS020 Capillary excluded/Run 09 and 11/'
            'Patient_26  16-26-12.JPG')"""

    """if run_num == "Run03":
        target = cv2.imread(
            'brain/HS026/Microscope images/'
            'Patient_61  12-49-16.JPG')
    if run_num == "Run05":
        target = cv2.imread(
            'brain/HS026/Microscope images/'
            'Patient_61  13-27-23.JPG')"""

    """if run_num == "Run01":
        target = cv2.imread(
            'brain/HS027/Microscope acquisitions/'
            'Patient_61  17-22-24.JPG')
    if run_num == "Run03":
        target = cv2.imread(
            'brain/HS027/Microscope acquisitions/'
            'Patient_61  17-35-35.JPG')
    if run_num == "Run05":
        target = cv2.imread(
            'brain/HS027/Microscope acquisitions/'
            'Patient_61  18-38-39.JPG')"""

    """if run_num == "Run03":
        target = cv2.imread(
            'brain/HS029/Images/'
            'Patient_73  12-56-25.JPG')
    if run_num == "Run07" or "Run08":
        target = cv2.imread(
            'brain/HS029/Images/'
            'Patient_73  13-50-37.JPG')"""

    """if run_num == "Run04":
        target = cv2.imread(
            'brain/HS019/Labelling data/Definitive segmentations/HS019/Run04-05/'
            '__2022-04-13_09-16-48_I.JPG')
    if run_num == "Run06":
        target = cv2.imread(
            'brain/HS019/Labelling data/Definitive segmentations/HS019/Run06-07/'
            '__2022-04-13_09-47-05_I.JPG')
    if run_num == "Run08":
        target = cv2.imread(
            'brain/HS019/Labelling data/Definitive segmentations/HS019/Run 08-09/'
            '__2022-04-13_10-45-26_I.JPG')"""
    if run_num == "Run05":
        target = cv2.imread(
            "brain/HS006/Labelling data/Definitive segmentations/HS006 Capillary excluded/Run 05-07/"
            "Patient_2_2021-07-28_09-58-29_I.JPG"
        )

    if run_num == "Run07" or run_num == "Run08":
        target = cv2.imread(
            "brain/HS006/Labelling data/Definitive segmentations/HS006 Capillary excluded/Run 08/"
            "Patient_2_2021-07-28_10-12-53_I.JPG"
        )
    if run_num == "Run11":
        target = cv2.imread(
            "brain/HS006/Labelling data/Definitive segmentations/HS006 Capillary excluded/Run 11/"
            "Patient_2_2021-07-28_11-39-52_I.JPG"
        )
    if run_num == "Run12":
        target = cv2.imread(
            "brain/HS006/Labelling data/Definitive segmentations/HS006 Capillary excluded/Run 12/"
            "Patient_2_2021-07-28_11-47-57_I.JPG"
        )
    if run_num == "Run13" or run_num == "Run14":
        target = cv2.imread(
            "brain/HS006/Labelling data/Definitive segmentations/HS006 Capillary excluded/Run 13/"
            "Patient_2_2021-07-28_11-48-01_I.JPG"
        )

    rows, cols, _ = target.shape

    rgb_flip = cv2.flip(target, 0)

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/target.png", rgb_flip)

    (bt, gt, rt) = cv2.split(rgb_flip)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/bt.png", bt)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/gt.png", gt)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/rt.png", rt)

    # fusion img 与 target reg
    regb, mergb, mab1 = registeration(
        bm,
        save_path + "/result_" + run_num[3:] + "/blue.png",
        save_path + "/result_" + run_num[3:] + "/bt.png",
        contrast=0.01,
        edge=20,
    )

    source = cv2.imread(save_path + "/result_" + run_num[3:] + "/green.png")
    regg = cv2.warpAffine(source, mab1, (cols, rows))

    source = cv2.imread(save_path + "/result_" + run_num[3:] + "/red.png")
    regr = cv2.warpAffine(source, mab1, (cols, rows))
    """regg, mergg, mag1 = registeration(gm, save_path+'/result_'+run_num[3:]+'/green.png', save_path+'/result_'+run_num[3:]+'/gt.png'
                                      ,contrast=0.01,edge=20)
    regr, mergr, mar1 = registeration(rm, save_path+'/result_'+run_num[3:]+'/red.png', save_path+'/result_'+run_num[3:]+'/rt.png'
                                      ,contrast=0.01,edge=20)"""

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue_reg.png", regb)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/green_reg.png", regg)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/red_reg.png", regr)

    # second reg
    # regb, mergb, mab2 = registeration("sift", save_path+'/result_'+run_num[3:]+'/blue_reg.png',
    # save_path + '/result_' + run_num[3:] + '/bt.png',contrast=0.01,edge=20)

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue_merg.png", mergb)

    b = cv2.imread(save_path + "/result_" + run_num[3:] + "/blue_reg.png", 0)
    g = cv2.imread(save_path + "/result_" + run_num[3:] + "/green_reg.png", 0)
    r = cv2.imread(save_path + "/result_" + run_num[3:] + "/red_reg.png", 0)

    """mb = cv2.imread(save_path + '/result_' + run_num[3:] + '/blue_merg.png', 0)
    mg = cv2.imread(save_path + '/result_' + run_num[3:] + '/green_merg.png', 0)
    mr = cv2.imread(save_path + '/result_' + run_num[3:] + '/red_merg.png', 0)"""

    rgb = cv2.merge((b, g, r))
    # rgb_merg = cv2.merge((mb, mg, mr))

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/im.png", rgb)
    # cv2.imwrite(save_path+'/result_'+run_num[3:]+'/rgb_merg.png', rgb_merg)

    # register all stacks
    """blue_target = save_path+'/result_'+run_num[3:]+'/blue_reg.png'
    green_target = save_path+'/result_' + run_num[3:] + '/green_reg.png'
    red_target = save_path+'/result_' + run_num[3:] + '/red_reg.png'

    register_img, m_img, rgb_m1 = registeration('sift', save_path + '/result_' + run_num[3:] + '/im_merg.png',
                                  save_path+'/result_'+run_num[3:]+'/target.png',contrast=0.02,edge=20)

    cv2.imwrite(save_path + '/result_' + run_num[3:] + '/im.png', register_img)"""

    """imb = cv2.imread(save_path + '/result_' + run_num[3:] + '/im.png')
    alpha = 2  # Contrast control
    #beta = 15  # Brightness control
    # call convertScaleAbs function
    sharpness_level = 9
    # Call the function to unblur/sharpen the image with the specified sharpness level
    adjustb = unblur_image(imb, sharpness_level)
    adjustb = cv2.convertScaleAbs(adjustb, alpha=alpha)#, beta=beta)

    cv2.imwrite(save_path + '/result_' + run_num[3:] + '/im.png', adjustb)"""
    """register_img, m_img, rgb_m2 = registeration('sift', save_path + '/result_' + run_num[3:] + '/im.png',
                                                save_path + '/result_' + run_num[3:] + '/target.png', contrast=0.02,
                                                edge=20)
    cv2.imwrite(save_path + '/result_' + run_num[3:] + '/im.png', register_img)"""

    rgb_mask(save_path + "/result_" + run_num[3:])

    resumePth = (
        "rf/model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth"  ## model for visualization
    )
    kernelSize = 7

    Transform = outil.Homography
    nbPoint = 4

    ## Loading model
    # Define Networks
    network = {
        "netFeatCoarse": model.FeatureExtractor(),
        "netCorr": model.CorrNeigh(kernelSize),
        "netFlowCoarse": model.NetFlowCoarse(kernelSize),
        "netMatch": model.NetMatchability(kernelSize),
    }

    for key in list(network.keys()):
        network[key]
        typeData = torch.FloatTensor

    # loading Network
    param = torch.load(resumePth, map_location=torch.device("cpu"))
    msg = "Loading pretrained model from {}".format(resumePth)
    # print(msg)

    for key in list(param.keys()):
        network[key].load_state_dict(param[key])
        network[key].eval()

    I1 = Image.open(save_path + "/result_" + run_num[3:] + "/im.png").convert("RGB")
    I2 = Image.open(save_path + "/result_" + run_num[3:] + "/mask.png").convert("RGB")

    nbScale = 7
    coarseIter = 10000
    coarsetolerance = 0.05
    minSize = I2.size[0]
    print(minSize)
    imageNet = True  # we can also use MOCO feature here
    scaleR = 1.2

    coarseModel = CoarseAlign(
        nbScale, coarseIter, coarsetolerance, "Homography", minSize, 1, True, imageNet, scaleR
    )

    coarseModel.setSource(I1)
    coarseModel.setTarget(I2)

    I2w, I2h = coarseModel.It.size
    featt = F.normalize(network["netFeatCoarse"](coarseModel.ItTensor))

    #### -- grid
    gridY = torch.linspace(-1, 1, steps=I2h).view(1, -1, 1, 1).expand(1, I2h, I2w, 1)
    gridX = torch.linspace(-1, 1, steps=I2w).view(1, 1, -1, 1).expand(1, I2h, I2w, 1)
    grid = torch.cat((gridX, gridY), dim=3)
    warper = tgm.HomographyWarper(I2h, I2w)

    bestPara, InlierMask = coarseModel.getCoarse(np.zeros((I2h, I2w)))
    bestPara = torch.from_numpy(bestPara).unsqueeze(0)

    flowCoarse = warper.warp_grid(bestPara)
    I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
    I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())

    I1_coarse_pil.save(save_path + "/result_" + run_num[3:] + "/mco.png")

    featsSample = F.normalize(network["netFeatCoarse"](I1_coarse))

    corr12 = network["netCorr"](featt, featsSample)
    flowDown8 = network["netFlowCoarse"](corr12, False)  ## output is with dimension B, 2, W, H

    flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode="bilinear")
    flowUp = flowUp.permute(0, 2, 3, 1)

    flowUp = flowUp + grid

    flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

    I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
    I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())

    I1_fine_pil.save(save_path + "/result_" + run_num[3:] + "/mfine.png")

    for i in range(25):
        # print('                    ',img_path+'/' + rpm + '/'+str(490 + 10 * i) + '.png')
        """registered, m = registeration('sift', img_path+'/' + rpm + '/'+str(460 + 10 * i) + '.png', blue_target)#registeration('sift', save_path+"/result_"+run_num[3:]+"/enhance_"+run_num + '/' + str(440 + 10 * i) + '.png', blue_target)
        registered = np.uint8(registered)
        m = np.uint8(m)

        reg_flip = cv2.flip(registered, 0)
        m_flip = cv2.flip(m, 0)"""
        source = cv2.imread(img_path + "/" + rpm + "/" + str(460 + 10 * i) + ".png")
        warp = cv2.warpAffine(source, mab1, (cols, rows))
        # warp = cv2.warpAffine(warp, rgb_m2, (cols, rows))
        registered = np.uint8(warp)
        # reg_flip = cv2.flip(registered, 0)

        cv2.imwrite(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(460 + 10 * i)
            + ".png",
            registered,
        )
        I1 = Image.open(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(460 + 10 * i)
            + ".png"
        ).convert("RGB")

        coarseModel.setSource(I1)
        I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
        I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
        I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())

        I1_fine_pil.save(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(460 + 10 * i)
            + ".png"
        )

        reg = cv2.imread(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(460 + 10 * i)
            + ".png"
        )
        reg_flip = cv2.flip(reg, 0)
        cv2.imwrite(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(460 + 10 * i)
            + ".png",
            reg_flip,
        )

    path = save_path + "/result_" + run_num[3:] + "/"
    run_num = run_num + "_reg/"

    b = cv2.imread(path + run_num + "490.png", 0)
    g = cv2.imread(path + run_num + "540.png", 0)
    r = cv2.imread(path + run_num + "600.png", 0)
    rgb = cv2.imread(path + "target.png")
    rgb_flip = cv2.flip(rgb, 0)
    rgb_merg = cv2.merge((b, g, r))
    merge = rgb_merg * 0.5 + rgb_flip * 0.5
    cv2.imwrite(path + run_num + "rgb.png", rgb_merg)
    cv2.imwrite(path + run_num + "merge.png", merge)
    cv2.imwrite(path + run_num + "target.png", rgb_flip)
