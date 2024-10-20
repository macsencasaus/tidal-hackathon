"""
Create the complete 1-Dimensional U-net model to be trained.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from keras import Model
from keras.api.activations import sigmoid
from keras.api.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    LeakyReLU,
    MaxPooling2D,
    concatenate,
)


def get_model(img_size: tuple = (256, 256), img_channels: int = 2) -> Model:
    """
    Creates and returns the U-net

    Args:
        img_size: dimension of image
        img_channel: number of channels (number of detectors)

    Returns:
        1D u-net
    """

    # -------------------------------------------------------------------------
    # Define the model's layers
    # -------------------------------------------------------------------------

    # intial number of channels that grows linearly with each U-net step
    layers_n = 32

    # input layer
    s = Input((*img_size, img_channels))

    # 1st U-net step (downscaling)
    c1 = Conv2D(layers_n, kernel_size=2, dilation_rate=1, strides=1, padding="same")(s)
    c1 = LeakyReLU()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(layers_n, kernel_size=2, dilation_rate=1, strides=1, padding="same")(c1)
    c1 = LeakyReLU()(c1)
    c1 = Dropout(0.1)(c1)
    p1 = MaxPooling2D(pool_size=2)(c1)

    # 2nd U-net step (downscaling)
    c2 = Conv2D(
        layers_n * 2, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(p1)
    c2 = LeakyReLU()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(
        layers_n * 2, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c2)
    c2 = LeakyReLU()(c2)
    c2 = Dropout(0.1)(c2)
    p2 = MaxPooling2D(pool_size=2)(c2)

    # 3rd U-net step (downscaling)
    c3 = Conv2D(
        layers_n * 3, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(p2)
    c3 = LeakyReLU()(c3)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(
        layers_n * 3, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c3)
    c3 = LeakyReLU()(c3)
    c3 = Dropout(0.1)(c3)
    p3 = MaxPooling2D(pool_size=2)(c3)

    # 4th U-net step (downscaling)
    c4 = Conv2D(
        layers_n * 4, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(p3)
    c4 = LeakyReLU()(c4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(
        layers_n * 4, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c4)
    c4 = LeakyReLU()(c4)
    c4 = Dropout(0.1)(c4)
    p4 = MaxPooling2D(pool_size=2)(c4)

    # 5th U-net step (downscaling)
    c5 = Conv2D(
        layers_n * 5, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(p4)
    c5 = LeakyReLU()(c5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(
        layers_n * 5, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c5)
    c5 = LeakyReLU()(c5)
    c5 = Dropout(0.1)(c5)
    p5 = MaxPooling2D(pool_size=2)(c5)

    # 6th U-net step (downscaling)
    c6 = Conv2D(
        layers_n * 6, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(p5)
    c6 = LeakyReLU()(c6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(
        layers_n * 6, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c6)
    c6 = LeakyReLU()(c6)
    c6 = Dropout(0.1)(c6)
    p6 = MaxPooling2D(pool_size=2)(c6)

    # 7th U-net step (downscaling)
    c7 = Conv2D(
        layers_n * 7, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(p6)
    c7 = LeakyReLU()(c7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(
        layers_n * 7, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c7)
    c7 = LeakyReLU()(c7)
    c7 = Dropout(0.1)(c7)

    # 8th U-net step (upscaling)
    u8 = Conv2DTranspose(layers_n * 6, kernel_size=2, strides=2, padding="same")(c7)
    u8 = concatenate([u8, c6])
    c8 = Conv2D(
        layers_n * 6, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(u8)
    c8 = LeakyReLU()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(
        layers_n * 6, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c8)
    c8 = LeakyReLU()(c8)

    # 9th U-net step (upscaling)
    u9 = Conv2DTranspose(layers_n * 5, kernel_size=2, strides=2, padding="same")(c8)
    u9 = concatenate([u9, c5])
    c9 = Conv2D(
        layers_n * 5, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(u9)
    c9 = LeakyReLU()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(
        layers_n * 5, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c9)
    c9 = LeakyReLU()(c9)

    # 10th U-net step (upscaling)
    u10 = Conv2DTranspose(layers_n * 4, kernel_size=2, strides=2, padding="same")(c9)
    u10 = concatenate([u10, c4])
    c10 = Conv2D(
        layers_n * 4, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(u10)
    c10 = LeakyReLU()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(
        layers_n * 4, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c10)
    c10 = LeakyReLU()(c10)

    # 11th U-net step (upscaling)
    u11 = Conv2DTranspose(layers_n * 3, kernel_size=2, strides=2, padding="same")(c10)
    u11 = concatenate([u11, c3])
    c11 = Conv2D(
        layers_n * 3, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(u11)
    c11 = LeakyReLU()(c11)
    c11 = Dropout(0.1)(c11)
    c11 = Conv2D(
        layers_n * 3, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c11)
    c11 = LeakyReLU()(c11)

    # 12th U-net step (Upscaling)
    u12 = Conv2DTranspose(layers_n * 2, kernel_size=2, strides=2, padding="same")(c11)
    u12 = concatenate([u12, c2])
    c12 = Conv2D(
        layers_n * 2, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(u12)
    c12 = LeakyReLU()(c12)
    c12 = Dropout(0.1)(c12)
    c12 = Conv2D(
        layers_n * 2, kernel_size=2, dilation_rate=1, strides=1, padding="same"
    )(c12)
    c12 = LeakyReLU()(c12)

    # 13th U-net step (upscaling)
    u13 = Conv2DTranspose(layers_n, kernel_size=2, strides=2, padding="same")(c12)
    u13 = concatenate([u13, c1])
    c13 = Conv2D(layers_n, kernel_size=2, dilation_rate=1, strides=1, padding="same")(
        u13
    )
    c13 = LeakyReLU()(c13)
    c13 = Dropout(0.1)(c13)
    c13 = Conv2D(layers_n, kernel_size=2, dilation_rate=1, strides=1, padding="same")(
        c13
    )
    c13 = LeakyReLU()(c13)

    # Final convolution bringing the channels back to 2
    outputs = Conv2D(img_channels, kernel_size=1)(c13)
    outputs = sigmoid(outputs)
    model = Model(inputs=[s], outputs=[outputs])

    return model


# -----------------------------------------------------------------------------
# MAIN CODE (TESTING GROUND TO PRINT MODEL SUMMARY)
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Start stopwatch
    script_start = time.time()

    print("Acquiring Model... ", end="", flush=True)
    model = get_model()
    print("Done!", flush=True)

    print("")
    print("Printing Model Summary... ", flush=True)
    print("")

    print(model.summary())

    print("Total Runtime: {:.1f}.".format(time.time() - script_start))
    print("")
