from __future__ import print_function

from feature.affine_ransac import Ransac
from feature.align_transform import Align
from feature.affine_transform import Affine
import argparse
import numpy as np
import tensorflow as tf

from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model
from keras.models import Sequential

from PIL import Image, ImageEnhance
from tqdm import tqdm
import os
import cv2


def ssim_loss(img1, img2):
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


def registeration(mode, source, moving):
    af = Affine()

    outlier_rate = 0.9
    A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate)

    K = 3
    idx = np.random.randint(0, pts_s.shape[1], (K, 1))
    A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

    rs = Ransac(K=3, threshold=1)

    residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

    A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)

    al = Align(source, moving, mode=mode, threshold=1)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ref", help="ref image name")
    # parser.add_argument("--moving", help="moving image name")
    parser.add_argument("-p", "--path", help="img path")
    parser.add_argument("-rpm", "--repeat_mode", help="orb, sift, akaze", default="sift")
    parser.add_argument("-bm", "--blue_mode", help="orb, sift, akaze", default="sift")
    parser.add_argument("-gm", "--green_mode", help="orb, sift, akaze", default="sift")
    parser.add_argument("-rm", "--red_mode", help="orb, sift, akaze", default="sift")

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

    path = "data/"
    # path_list = os.listdir(path)
    path_list = ["Run111", "Run121", "Run131", "Run141"]
    X = []

    for i in tqdm(range(len(path_list))):
        for img in os.listdir(path + path_list[i]):
            c = cv2.imread(path + path_list[i] + "/" + img, 0)
            c = cv2.resize(c, (224, 224))
            c = c.astype(np.float32)
            X.append(c)

    X = np.array(X).reshape(-1, 224, 224, 1)

    Autoencoder = recon_model(input_shape=(224, 224, 1))
    Autoencoder.compile(optimizer="adam", loss=ssim_loss, metrics=["accuracy"])

    hist = Autoencoder.fit(X, X, epochs=30)

    Autoencoder.save("model/HS014_Run045.h5")

    # fusion
    Autoencoder = recon_model(input_shape=(None, None, 1))
    Autoencoder.load_weights("model/HS014_Run045.h5")
    Autoencoder.compile(optimizer="adam", loss=ssim_loss, metrics=["accuracy"])

    imgI = cv2.imread("brain/HS009/Run04/image_sample_630nm_2.png", 0)

    w, h = imgI.shape[0], imgI.shape[1]

    encoder = Model(inputs=Autoencoder.layers[0].input, outputs=Autoencoder.layers[7].output)

    decoder = Sequential()
    for layer in Autoencoder.layers[8:]:
        decoder.add(layer)

    b_list = []
    g_list = []
    r_list = []

    path = save_path + "/result_" + run_num[3:] + "/" + run_num

    for i in range(8):
        c = cv2.imread(path + "/" + str(460 + 10 * i) + ".png", 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        b_list.append(c)

    for i in range(8):
        c = cv2.imread(path + "/" + str(540 + 10 * i) + ".png", 0)
        # c = cv2.resize(c, (224, 224))
        c = c.astype(np.float32)
        g_list.append(c)

    for i in range(9):
        c = cv2.imread(path + "/" + str(620 + 10 * i) + ".png", 0)
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
    fuse_R = decoder.predict(fuse_r)

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue.png", fuse_B[0, :, :, 0])
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/green.png", fuse_G[0, :, :, 0])
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/red.png", fuse_R[0, :, :, 0])

    # sharp
    b = cv2.imread(save_path + "/result_" + run_num[3:] + "/blue.png", 0)
    g = cv2.imread(save_path + "/result_" + run_num[3:] + "/green.png", 0)
    r = cv2.imread(save_path + "/result_" + run_num[3:] + "/red.png", 0)

    ib = cv2.imread(save_path + "/result_" + run_num[3:] + "/blue.png", flags=cv2.IMREAD_COLOR)
    ig = cv2.imread(save_path + "/result_" + run_num[3:] + "/green.png", flags=cv2.IMREAD_COLOR)
    ir = cv2.imread(save_path + "/result_" + run_num[3:] + "/red.png", flags=cv2.IMREAD_COLOR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    ib_sharp = cv2.filter2D(src=ib, ddepth=-1, kernel=kernel)
    ig_sharp = cv2.filter2D(src=ig, ddepth=-1, kernel=kernel)
    ir_sharp = cv2.filter2D(src=ir, ddepth=-1, kernel=kernel)

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue.png", ib_sharp)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/green.png", ig_sharp)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/red.png", ir_sharp)

    factor = 25
    imb = Image.open(save_path + "/result_" + run_num[3:] + "/blue.png")
    enhancer = ImageEnhance.Sharpness(imb)
    im_s_1 = enhancer.enhance(factor)
    im_s_1.save(save_path + "/result_" + run_num[3:] + "/blue.png")

    img = Image.open(save_path + "/result_" + run_num[3:] + "/green.png")
    enhancer = ImageEnhance.Sharpness(img)
    im_s_1 = enhancer.enhance(factor)
    im_s_1.save(save_path + "/result_" + run_num[3:] + "/green.png")

    imr = Image.open(save_path + "/result_" + run_num[3:] + "/red.png")
    enhancer = ImageEnhance.Sharpness(imr)
    im_s_1 = enhancer.enhance(factor)
    im_s_1.save(save_path + "/result_" + run_num[3:] + "/red.png")

    if run_num == "Run02" or run_num == "Run03":
        target = cv2.imread(
            "brain/HS014/2022-01-26 13-53-43 Patient_1/Images/Patient_1_2022-01-26_14-17-37_I.JPG"
        )
    if run_num == "Run04" or run_num == "Run05":
        target = cv2.imread(
            "brain/HS014/2022-01-26 13-53-43 Patient_1/Images/Patient_1_2022-01-26_14-53-07_I.JPG"
        )

    rgb_flip = cv2.flip(target, 0)
    (bt, gt, rt) = cv2.split(rgb_flip)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/target.png", rgb_flip)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/bt.png", bt)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/gt.png", gt)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/rt.png", rt)

    regb, mergb = registeration(
        bm,
        save_path + "/result_" + run_num[3:] + "/blue.png",
        save_path + "/result_" + run_num[3:] + "/bt.png",
    )
    regg, mergg = registeration(
        gm,
        save_path + "/result_" + run_num[3:] + "/green.png",
        save_path + "/result_" + run_num[3:] + "/gt.png",
    )
    regr, mergr = registeration(
        rm,
        save_path + "/result_" + run_num[3:] + "/red.png",
        save_path + "/result_" + run_num[3:] + "/rt.png",
    )

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue_reg.png", regb)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/green_reg.png", regg)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/red_reg.png", regr)

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/blue_merg.png", mergb)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/green_merg.png", mergg)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/red_merg.png", mergr)

    b = cv2.imread(save_path + "/result_" + run_num[3:] + "/blue_reg.png", 0)
    g = cv2.imread(save_path + "/result_" + run_num[3:] + "/green_reg.png", 0)
    r = cv2.imread(save_path + "/result_" + run_num[3:] + "/red_reg.png", 0)

    mb = cv2.imread(save_path + "/result_" + run_num[3:] + "/blue_merg.png", 0)
    mg = cv2.imread(save_path + "/result_" + run_num[3:] + "/green_merg.png", 0)
    mr = cv2.imread(save_path + "/result_" + run_num[3:] + "/red_merg.png", 0)

    rgb = cv2.merge((b, g, r))
    rgb_merg = cv2.merge((mb, mg, mr))

    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/rgb.png", rgb)
    cv2.imwrite(save_path + "/result_" + run_num[3:] + "/rgb_merg.png", rgb_merg)

    # register all stacks
    blue_target = save_path + "/result_" + run_num[3:] + "/blue_reg.png"
    green_target = save_path + "/result_" + run_num[3:] + "/green_reg.png"
    red_target = save_path + "/result_" + run_num[3:] + "/red_reg.png"

    for i in range(6):
        # print('                    ',img_path+'/' + rpm + '/'+str(490 + 10 * i) + '.png')
        registered, m = registeration(
            "sift", img_path + "/" + rpm + "/" + str(490 + 10 * i) + ".png", blue_target
        )  # registeration('sift', save_path+"/result_"+run_num[3:]+"/enhance_"+run_num + '/' + str(440 + 10 * i) + '.png', blue_target)
        registered = np.uint8(registered)
        m = np.uint8(m)

        reg_flip = cv2.flip(registered, 0)
        m_flip = cv2.flip(m, 0)

        cv2.imwrite(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(490 + 10 * i)
            + ".png",
            reg_flip,
        )
        # cv2.imwrite(save_path + "/result_" + run_num[3:] + "/" + run_num + "_merg/" + str(460 + 10 * i) + "_merg.png", m_flip)

    for i in range(8):
        registered, m = registeration(
            "sift", img_path + "/" + rpm + "/" + str(540 + 10 * i) + ".png", green_target
        )  # save_path+"/result_"+run_num[3:]+"/enhance_"+run_num + '/' + str(540 + 10 * i) + '.png', green_target)
        registered = np.uint8(registered)
        m = np.uint8(m)

        reg_flip = cv2.flip(registered, 0)
        m_flip = cv2.flip(m, 0)

        cv2.imwrite(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(540 + 10 * i)
            + ".png",
            reg_flip,
        )
        # cv2.imwrite(save_path + "/result_" + run_num[3:] + "/" + run_num + "_merg/" + str(540 + 10 * i) + "_merg.png",
        # m_flip)

    for i in range(10):
        # print('                    ', img_path + '/' + rpm + '/' + str(620 + 10 * i) + '.png')
        registered, m = registeration(
            "sift", img_path + "/" + rpm + "/" + str(620 + 10 * i) + ".png", red_target
        )  # save_path+"/result_"+run_num[3:]+"/enhance_"+run_num + '/' + str(620 + 10 * i) + '.png', red_target)
        registered = np.uint8(registered)
        m = np.uint8(m)

        reg_flip = cv2.flip(registered, 0)
        m_flip = cv2.flip(m, 0)

        cv2.imwrite(
            save_path
            + "/result_"
            + run_num[3:]
            + "/"
            + run_num
            + "_reg/"
            + str(620 + 10 * i)
            + ".png",
            reg_flip,
        )
        # cv2.imwrite(save_path + "/result_" + run_num[3:] + "/" + run_num + "_merg/" + str(620 + 10 * i) + "_merg.png",
        # m_flip)
