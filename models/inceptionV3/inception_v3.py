# -*- coding: utf-8 -*-
"""Inception V3 model for TensorFlow 2.0.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""

import tensorflow as tf

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

class ConvBNRelu(tf.keras.Model):

    def __init__(self, out_channels, strides=1, kernel_size=3, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size,
            strides=strides, padding=padding, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(axis=3, scale=False)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x



class InceptionV3(tf.keras.Model):

    def __init__(self, include_top=True, weights=None, pooling=None, classes=1000):
        super(InceptionV3, self).__init__()

        self.include_top = include_top
        self.model_weights = weights
        self.pooling = pooling
        self.classes = classes

        self.Conv2d_1a_3x3 = ConvBNRelu(32, kernel_size=3, strides=(2, 2), padding='valid')
        self.Conv2d_2a_3x3 = ConvBNRelu(32, kernel_size=3, padding='valid')
        self.Conv2d_2b_3x3 = ConvBNRelu(64, kernel_size=3)
        self.Conv2d_3b_1x1 = ConvBNRelu(80, kernel_size=1)
        self.Conv2d_4a_3x3 = ConvBNRelu(192, kernel_size=3)

        self.Mixed_5b = InceptionA(pool_features=32)
        self.Mixed_5c = InceptionA(pool_features=64)
        self.Mixed_5d = InceptionA(pool_features=64)

        self.Mixed_6a = InceptionB()

        self.Mixed_6b = InceptionC(channels_7x7=128)
        self.Mixed_6c = InceptionC(channels_7x7=160)
        self.Mixed_6d = InceptionC(channels_7x7=160)
        self.Mixed_6e = InceptionC(channels_7x7=192)

        self.Mixed_7a = InceptionD()

        self.Mixed_7b = InceptionE()
        self.Mixed_7c = InceptionE()

        if self.include_top:
            self.fc = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(inputs)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)

        if self.include_top:
            x = GlobalAveragePooling2D()(x)
            x = self.fc(x)
        else:
            if self.pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        return x


class InceptionA(tf.keras.Model):

    def __init__(self, pool_features):
        super(InceptionA, self).__init__()
        # input channels: 192
        # mixed 1: 35 x 35 x 256
        self.branch1x1 = ConvBNRelu(64, kernel_size=1)

        self.branch5x5_1 = ConvBNRelu(48, kernel_size=1)
        self.branch5x5_2 = ConvBNRelu(64, kernel_size=5)

        self.branch3x3dbl_1 = ConvBNRelu(64, kernel_size=1)
        self.branch3x3dbl_2 = ConvBNRelu(96, kernel_size=3)
        self.branch3x3dbl_3 = ConvBNRelu(96, kernel_size=3)

        self.branch_pool = ConvBNRelu(pool_features, kernel_size=1)

    def call(self, inputs, training=None, mask=None):
        branch1x1 = self.branch1x1(inputs)

        branch5x5 = self.branch5x5_1(inputs)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(inputs)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_pool = self.branch_pool(branch_pool)

        outputs = tf.keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=-1)
        return outputs


class InceptionB(tf.keras.Model):

    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch3x3 = ConvBNRelu(384, kernel_size=3, strides=(2, 2), padding='valid')

        self.branch3x3dbl_1 = ConvBNRelu(64, kernel_size=1)
        self.branch3x3dbl_2 = ConvBNRelu(96, kernel_size=3)
        self.branch3x3dbl_3 = ConvBNRelu(96, kernel_size=3, strides=(2, 2), padding='valid')


    def call(self, inputs, training=None, mask=None):
        branch3x3 = self.branch3x3(inputs)

        branch3x3dbl = self.branch3x3dbl_1(inputs)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(inputs)

        outputs = tf.keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=-1)
        return outputs


class InceptionC(tf.keras.Model):

    def __init__(self, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = ConvBNRelu(192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = ConvBNRelu(c7, kernel_size=1)
        self.branch7x7_2 = ConvBNRelu(c7, kernel_size=(1, 7))
        self.branch7x7_3 = ConvBNRelu(192, kernel_size=(7, 1))


        self.branch7x7dbl_1 = ConvBNRelu(c7, kernel_size=(1, 1))
        self.branch7x7dbl_2 = ConvBNRelu(c7, kernel_size=(7, 1))
        self.branch7x7dbl_3 = ConvBNRelu(c7, kernel_size=(1, 7))
        self.branch7x7dbl_4 = ConvBNRelu(c7, kernel_size=(7, 1))
        self.branch7x7dbl_5 = ConvBNRelu(192, kernel_size=(7, 1))

        self.branch_pool = ConvBNRelu(192, kernel_size=1)

    def call(self, inputs, training=None, mask=None):
        branch1x1 = self.branch1x1(inputs)

        branch7x7 = self.branch7x7_1(inputs)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(inputs)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_pool = self.branch_pool(branch_pool)

        outputs = tf.keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=-1)
        return outputs


class InceptionD(tf.keras.Model):

    def __init__(self):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = ConvBNRelu(192, kernel_size=1)
        self.branch3x3_2 = ConvBNRelu(320, kernel_size=3, strides=(2, 2), padding='valid')

        self.branch7x7x3_1 = ConvBNRelu(192, kernel_size=1)
        self.branch7x7x3_2 = ConvBNRelu(192, kernel_size=(1, 7))
        self.branch7x7x3_3 = ConvBNRelu(192, kernel_size=(7, 1))
        self.branch7x7x3_4 = ConvBNRelu(192, kernel_size=3, strides=(2, 2), padding='valid')

    def call(self, inputs, training=None, mask=None):
        branch3x3 = self.branch3x3_1(inputs)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(inputs)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(inputs)

        outputs = tf.keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=-1)
        return outputs


class InceptionE(tf.keras.Model):

    def __init__(self):
        super(InceptionE, self).__init__()
        self.branch1x1 = ConvBNRelu(320, kernel_size=1)

        self.branch3x3_1 = ConvBNRelu(384, kernel_size=1)
        self.branch3x3_2a = ConvBNRelu(384, kernel_size=(1, 3))
        self.branch3x3_2b = ConvBNRelu(384, kernel_size=(3, 1))

        self.branch3x3dbl_1 = ConvBNRelu(448, kernel_size=1)
        self.branch3x3dbl_2 = ConvBNRelu(384, kernel_size=3)
        self.branch3x3dbl_3a = ConvBNRelu(384, kernel_size=(1, 3))
        self.branch3x3dbl_3b = ConvBNRelu(384, kernel_size=(3, 1))

        self.branch_pool = ConvBNRelu(192, kernel_size=1)


    def call(self, inputs, training=None, mask=None):
        branch1x1 = self.branch1x1(inputs)

        branch3x3 = self.branch3x3_1(inputs)
        branch3x3 = [
                        self.branch3x3_2a(branch3x3),
                        self.branch3x3_2b(branch3x3)
                    ]
        branch3x3 = tf.keras.layers.concatenate(branch3x3, axis=-1)


        branch3x3dbl = self.branch3x3dbl_1(inputs)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl =  [
                            self.branch3x3dbl_3a(branch3x3dbl),
                            self.branch3x3dbl_3b(branch3x3dbl)
                        ]
        branch3x3dbl = tf.keras.layers.concatenate(branch3x3dbl, axis=-1)


        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_pool = self.branch_pool(branch_pool)

        outputs = tf.keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=-1)
        return outputs



if __name__ == '__main__':
    model = InceptionV3(include_top=False)
    # model = ConvBNRelu(32, kernel_size=3, strides=2, padding='valid')
    # model = InceptionA(pool_features=64)
    # model = InceptionB()
    # model = InceptionC(128)
    # model = InceptionD()
    # model = InceptionE()

    # image_size = 299
    # dummy_x = tf.zeros((1, image_size, image_size, 3))

    # model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
    #          loss=tf.keras.losses.categorical_crossentropy,
    #          metrics=['accuracy'])

    # model._set_inputs(dummy_x)
    model.build((None, 299, 299, 3))
    # model.load_weights(r'/Users/linrenhong/Documents/工研院/Inception_V3/pretrained_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model.summary()


