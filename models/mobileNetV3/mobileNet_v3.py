import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(input, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    input_size = tf.keras.backend.int_shape(input)[1:3]


    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class ConvBNRelu6(tf.keras.Model):

    def __init__(self, out_channels, strides=1, kernel_size=3, padding='same'):
        super(ConvBNRelu6, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
                                            out_channels,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            padding=padding,
                                            use_bias=False,
                                            name='Conv1'
                                          )

        self.bn = tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        epsilon=1e-3,
                                                        momentum=0.999,
                                                        name='bn_Conv1'
                                                    )
        # self.Hswish = Hswish()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu6(x, name='Conv1_relu')
        # x = self.Hswish(x)
        return x


class Conv_1x1_BNRelu6(tf.keras.Model):

    def __init__(self, out_channels, padding='valid'):
        super(Conv_1x1_BNRelu6, self).__init__()
        self.conv = tf.keras.layers.Conv2D(
                                            filters=out_channels,
                                            kernel_size=1,
                                            strides=1,
                                            padding=padding,
                                            use_bias=False,
                                            name='Conv_1'
                                          )

        self.bn = tf.keras.layers.BatchNormalization(
                                                        axis=-1,
                                                        epsilon=1e-3,
                                                        momentum=0.999,
                                                        name='bn_Conv_1'
                                                    )

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu6(x, name='out_relu')

        return x


class Hswish(tf.keras.layers.Layer):
    def __init__(self):
        super(Hswish, self).__init__()

    def call(self, inputs, training=None, mask=None):
        return inputs * tf.nn.relu6(inputs + 3.) / 6

if __name__ == '__main__':
    image_size = 224
    dummy_x = tf.zeros((1, image_size, image_size, 3))
    model = ConvBNRelu6(16)
    model.build((None, 224, 224, 3))
    # model._set_inputs(dummy_x)
    model.summary()

