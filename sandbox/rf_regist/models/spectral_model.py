import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, ImageEnhance

from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model
from keras.models import Sequential


# constants
WEIGHTS_PATH = "weights/spectral_encoder.h5"
KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
FACTOR = 25


def ssim_loss(img1, img2):
    # SSIM
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


# DECODER
def get_encoder_decoder():
    model = recon_model(input_shape=(None, None, 1))
    model.load_weights(WEIGHTS_PATH)
    model.compile(optimizer="adam", loss=ssim_loss, metrics=["accuracy"])

    encoder = Model(inputs=model.layers[0].input, outputs=model.layers[7].output)

    decoder = Sequential()
    for layer in model.layers[8:]:
        decoder.add(layer)

    return encoder, decoder


def get_inferred_image(images):
    w, h = images[0].shape
    encoder, decoder = get_encoder_decoder()

    feature_list = []
    for image in images:
        image = np.array(image).reshape(-1, w, h, 1)
        image_feature = encoder.predict(image, verbose=0)
        feature_list.append(image_feature)

    fused_feature = sum(feature_list) / len(feature_list)
    target_image = decoder.predict(fused_feature, verbose=0)

    # enchance image
    target_image = target_image.reshape(w, h)
    target_image = target_image.astype(np.uint8)
    # target_image = cv2.filter2D(target_image, ddepth=-1, kernel=KERNEL)
    # target_image = Image.fromarray(target_image)
    # enhancer = ImageEnhance.Sharpness(target_image)
    # target_image = enhancer.enhance(FACTOR)

    return target_image
