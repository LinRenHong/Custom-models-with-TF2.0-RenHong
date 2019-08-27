import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

# Option
is_show_input_shape = True
is_show_block_structure = False

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

    def __init__(self, out_channels, kernel_size=3, strides=1, padding='same'):
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

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu6(x, name='Conv1_relu')

        if is_show_input_shape:
            print("After Conv_bn_relu6 x: {}".format(x.shape))
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

        if is_show_input_shape:
            print("After Conv_1x1_bn_relu6 x: {}".format(x.shape))
        return x


class InvertedResidual(tf.keras.Model):
    def __init__(self, input_channels, expansion, stride, alpha, filters, block_id):
        super(InvertedResidual, self).__init__()
        self.expansion = expansion
        self.stride = stride
        self.alpha = alpha
        self.filters = filters
        self.block_id = block_id

        self.pointwise_conv_filters = int(filters * alpha)
        self.pointwise_filters = _make_divisible(self.pointwise_conv_filters, 8)
        self.prefix = 'block_{}_'.format(block_id)

        if block_id:
            # Expand
            self.conv_expand = tf.keras.layers.Conv2D(
                                                        filters=expansion * input_channels,
                                                        kernel_size=1,
                                                        padding='same',
                                                        use_bias=False,
                                                        activation=None,
                                                        name=self.prefix + 'expand'
                                                     )
            # axis = -1 when 'channels_last'
            self.bn_expand = tf.keras.layers.BatchNormalization(
                                                                axis=-1,
                                                                epsilon=1e-3,
                                                                momentum=0.999,
                                                                name=self.prefix + 'expand_BN'
                                                               )

        else:
            self.prefix = 'expanded_conv_'

        # Depthwise
        self.conv_depthwise = tf.keras.layers.DepthwiseConv2D(
                                                                kernel_size=3,
                                                                strides=stride,
                                                                activation=None,
                                                                use_bias=False,
                                                                padding='same' if stride == 1 else 'valid',
                                                                name=self.prefix + 'depthwise'
                                                             )
        self.bn_depthwise = tf.keras.layers.BatchNormalization(
                                                                axis=-1,
                                                                epsilon=1e-3,
                                                                momentum=0.999,
                                                                name=self.prefix + 'depthwise_BN'
                                                              )

        # Project
        self.conv_project = tf.keras.layers.Conv2D(
                                                    filters=self.pointwise_filters,
                                                    kernel_size=1,
                                                    padding='same',
                                                    use_bias=False,
                                                    activation=None,
                                                    name=self.prefix + 'project'
                                                  )

        self.bn_project = tf.keras.layers.BatchNormalization(
                                                                axis=-1,
                                                                epsilon=1e-3,
                                                                momentum=0.999,
                                                                name=self.prefix + 'project_BN'
                                                            )



    def call(self, inputs, training=None, mask=None):

        x = inputs
        self.input_channels = tf.keras.backend.int_shape(inputs)[-1]

        if self.block_id:
            x = self.conv_expand(x)
            x = self.bn_expand(x)
            # x = tf.keras.layers.ReLU(6., name=self.prefix + 'expand_relu')(x)
            x = tf.nn.relu6(x, name=self.prefix + 'expand_relu')

        if self.stride == 2:
            x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
                                 name=self.prefix + 'pad')(x)

        x = self.conv_depthwise(x)
        x = self.bn_depthwise(x)
        x = tf.nn.relu6(x, name=self.prefix + 'depthwise_relu')

        x = self.conv_project(x)
        x = self.bn_project(x)

        if self.input_channels == self.pointwise_filters and self.stride == 1:
            if is_show_input_shape:
                print("After InvertedResidual x: {} (stride = 1)".format(x.shape))

            return tf.keras.layers.Add(name=self.prefix + 'add')([inputs, x])

        if is_show_input_shape:
            print("After InvertedResidual x: {}".format(x.shape))

        return x


class MobileNetV2(tf.keras.Model):
    def __init__(self, alpha=1.0, include_top=True, weights=None, pooling=None, classes=1000):
        super(MobileNetV2, self).__init__()

        self.alpha = alpha
        self.include_top = include_top
        self.model_weights = weights
        self.pooling = pooling
        self.classes = classes
        self.first_block_filters = _make_divisible(32 * alpha, 8)
        self.input_channel = 32
        self.interverted_residual_setting = [
                                                # t, c, n, s
                                                [1, 16, 1, 1],
                                                [6, 24, 2, 2],
                                                [6, 32, 3, 2],
                                                [6, 64, 4, 2],
                                                [6, 96, 3, 1],
                                                [6, 160, 3, 2],
                                                [6, 320, 1, 1]
                                            ]


        self.init_conv = ConvBNRelu6(self.first_block_filters, kernel_size=3, strides=(2, 2), padding='valid')
        self.features = []


        block_idx = 0

        for t, c, n, s in self.interverted_residual_setting:
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channels=self.input_channel, filters=c, alpha=alpha, stride=s,
                                                            expansion=t, block_id=block_idx))
                else:
                    self.features.append(InvertedResidual(input_channels=self.input_channel, filters=c, alpha=alpha, stride=1,
                                                            expansion=t, block_id=block_idx))
                block_idx += 1
                self.input_channel = c

        if alpha > 1.0:
            self.last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            self.last_block_filters = 1280

        self.out_conv = Conv_1x1_BNRelu6(self.last_block_filters)

        if self.include_top:
            self.fc = tf.keras.layers.Dense(classes, activation='softmax',
                                            use_bias=True, name='Logits')




    def call(self, inputs, training=None, mask=None):

        x = inputs
        x = tf.keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
                             name='Conv1_pad')(x)
        x = self.init_conv(x)

        ### print structure ###
        if is_show_block_structure:
            self.init_conv.summary()

        for layer in self.features:
            x = layer(x)
            ### print structure ###
            if is_show_block_structure:
                layer.summary()

        x = self.out_conv(x)

        ### print structure ###
        if is_show_block_structure:
            self.out_conv.summary()


        if self.include_top:
            x = GlobalAveragePooling2D()(x)
            x = self.fc(x)
        else:
            if self.pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        if is_show_input_shape:
            print("After MobileNet_V2 x: {}".format(x.shape))

        return x


if __name__ == '__main__':
    image_size = 224
    dummy_x = tf.zeros((1, image_size, image_size, 3))
    print(tf.keras.backend.int_shape(dummy_x))
    print(tf.keras.backend.image_data_format())
    print(tf.keras.backend.int_shape(dummy_x)[1:3])
    # print(correct_pad((224,224), 3))
    # model = InvertedResidual(input_channels=32, stride=1, filters=16, block_id=0, expansion=1, alpha=1.0)
    model = MobileNetV2(include_top=True)
    model.build((None, 224, 224, 3))
    # model._set_inputs(dummy_x)
    model.summary()
