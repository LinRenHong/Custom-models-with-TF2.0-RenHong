import tensorflow as tf


from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

if __name__ == '__main__':
    model = MobileNetV2(include_top=False, weights=None)
    model.summary()