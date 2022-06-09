from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import VGG19


# from multiscale_unet import *


def uvgg19(input_size):
    if input_size[-1] == 1:
        input_size = (input_size[0], input_size[1], 3)
    vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_size, pooling=None)
    encoder = Model(vgg19.input, vgg19.get_layer("block5_conv4").output, name="encoder")

    d_i = Input(shape=(encoder.output.shape[1:]), name='decoder_input')
    block5_up = UpSampling2D(size=(2, 2), name="block5_up")(d_i)

    block4_1_conv0 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv0")(
        block5_up)
    block4_merge = Concatenate(axis=-1, name="block4_merge")([vgg19.get_layer("block4_conv4").output, block4_1_conv0])
    block4_1_conv1 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv1")(
        block4_merge)
    block4_1_conv2 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv2")(
        block4_1_conv1)
    block4_1_conv3 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv3")(
        block4_1_conv2)
    block4_1_conv4 = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', name="block4_1_conv4")(
        block4_1_conv3)
    block4_up = UpSampling2D(size=(2, 2), name="block4_up")(block4_1_conv4)

    block3_1_conv0 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv0")(
        block4_up)
    block3_merge = Concatenate(axis=-1, name="block3_merge")([vgg19.get_layer("block3_conv4").output, block3_1_conv0])
    block3_1_conv1 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv1")(
        block3_merge)
    block3_1_conv2 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv2")(
        block3_1_conv1)
    block3_1_conv3 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv3")(
        block3_1_conv2)
    block3_1_conv4 = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', name="block3_1_conv4")(
        block3_1_conv3)
    block3_up = UpSampling2D(size=(2, 2), name="block3_up")(block3_1_conv4)

    block2_1_conv0 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_1_conv0")(
        block3_up)
    block2_merge = Concatenate(axis=-1, name="block2_merge")([vgg19.get_layer("block2_conv2").output, block2_1_conv0])
    block2_1_conv1 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_1_conv1")(
        block2_merge)
    block2_1_conv2 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name="block2_1_conv2")(
        block2_1_conv1)
    block2_up = UpSampling2D(size=(2, 2), name="block2_up")(block2_1_conv2)

    block1_1_conv0 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_1_conv0")(
        block2_up)
    block1_merge = Concatenate(axis=-1, name="block1_merge")([vgg19.get_layer("block1_conv2").output, block1_1_conv0])
    block1_1_conv1 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_1_conv1")(
        block1_merge)
    block1_1_conv2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name="block1_1_conv2")(
        block1_1_conv1)
    block1_1_conv3 = Conv2D(filters=2, kernel_size=3, padding='same', activation='relu', name="prob_maps")(
        block1_1_conv2)
    output = Conv2D(filters=1, kernel_size=1, name="output")(
        block1_1_conv3)
    output_activation = Activation('sigmoid')(output)

    decoder = Model([vgg19.input, d_i], output_activation, name="decoder")

    decoder_output = decoder([vgg19.input, encoder(encoder.input)])
    model = Model(encoder.input, decoder_output, name=vgg19.name)

    return model
