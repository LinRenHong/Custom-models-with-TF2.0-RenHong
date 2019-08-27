# -*- coding: utf-8 -*-

import tensorflow as tf

# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from models.inception.inception_v3_sequential import InceptionV3
from models.inceptionV3.inception_v3 import InceptionV3
from models.mobileNetV2.mobileNet_v2 import MobileNetV2


from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D


def setup_to_transfer_learning(model, base_model):
	for layer in base_model.layers:
		layer.trainable = False
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_fine_tune(model, base_model):
	GAP_LAYER = 17 # max_pooling_2d_2
	for layer in base_model.layers[:GAP_LAYER+1]:
		layer.trainable = False
	for layer in base_model.layers[GAP_LAYER+1:]:
		layer.trainable = True
	model.compile(optimizer=tf.keras.optimizers.Adagrad(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


def CustomSequentialInception(num_classes, input_tensor=None, input_size=[299,299],
                    load_model_path='imagenet', 
                    pooling=None, dropout_rate=0.2):
                    
    input_size.append(3)

    base_model = InceptionV3( 
                                include_top=False,
                                weights=load_model_path,
                                input_tensor=input_tensor,
                                input_shape=input_size,
                                pooling=pooling,
                            )
    # if include_top:
    #     base_model.layers.pop()
    #     base_model.outputs = [base_model.layers[-1].output]
    #     base_model.layers[-1
    #     ].outbound_nodes = []

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes, activation='relu')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.summary()
    # setup_to_transfer_learning(model, base_model)
    setup_to_fine_tune(model, base_model)
    return model


class ModifiedInception(InceptionV3):

    def __init__(self, num_classes=1000, pooling=None):
        super(ModifiedInception, self).__init__(include_top=True, classes=num_classes, pooling=pooling)

    def call(self, inputs, training=None, mask=None, end_at="logits", add_inbetween_endpoints=False):

        end_points = {"4a" : None, "5a" : None, "6e" : None, "7c" : None, "fc" : None}

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

        if add_inbetween_endpoints or end_at=="4a":
            end_points["4a"] = x
            if end_at == "4a":
                return end_points

        # 71 x 71 x 192
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)

        if add_inbetween_endpoints or end_at=="5d":
            end_points["5d"] = x
            if end_at == "5d":
                return end_points

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

        if add_inbetween_endpoints or end_at=="6e":
            end_points["6e"] = x
            if end_at == "6e":
                return end_points

        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)

        if add_inbetween_endpoints or end_at=="7c":
            end_points["7c"] = x
            if end_at == "7c":
                return end_points

        # 8 x 8 x 2048
        kernel_size = x.shape[1]
        x = AveragePooling2D(kernel_size, padding='same')(x)
        # 1 x 1 x 2048
        x = layers.Dropout(0.2)(x)
        # 1 x 1 x 2048
        x = tf.keras.layers.Flatten()(x)

        if add_inbetween_endpoints or end_at=="features":
            end_points["features"] = x
            if end_at == "features":
                return end_points

        # 2048
        x = self.fc(x)

        end_points["logits"] = x

        return end_points

class CustomInceptionV3(tf.keras.Model):

    def __init__(self,
                 num_classes,
                 input_size,
                 re_train=True,
                 dropout_rate=0.2,):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.incepton_v3 = ModifiedInception()

        self.classifier = tf.keras.Sequential   ([
                                                    layers.Dropout(dropout_rate),
                                                    layers.Dense(num_classes, activation='softmax')
                                                ])
        self.incepton_v3.fc = self.classifier

    def call(self, inputs, training=None, mask=None):
        end_points = self.incepton_v3(inputs)

        print("Output shape: {}".format(end_points["logits"].shape))

        return end_points



if __name__ == "__main__":
    # base_model = InceptionV3(weights=None, include_top=True)
    # base_model.summary()
    # path = r'/Users/linrenhong/Documents/工研院/Inception_V3/pretrained_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # model = CustomSequentialInception(num_classes=1000, input_size=[299, 299])
    # print(type(model))
    # print(Sequential([layers.Dense(3, activation='softmax')]))
    # print(type(model.output))
    # plot_model(model, 'myModel.png')

    # model = InceptionV3()
    # model = ModifiedInception(include_top=False)
    model = CustomInceptionV3(30, 256, dropout_rate=0.8)
    # model = MobileNetV2()
    image_size = 299
    dummy_x = tf.zeros((1, image_size, image_size, 3))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

    model._set_inputs(dummy_x)
    model.build((None, 299, 299, 3))
    model.summary()
    


