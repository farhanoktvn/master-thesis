import tensorflow as tf

from keras.layers import Input, Conv2D, Concatenate
from keras.models import Model
from keras.models import Sequential


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
