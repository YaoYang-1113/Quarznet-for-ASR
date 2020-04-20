from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
set_session(tf.Session(config=config))
from quartznet import *
from train_quartzutils import train_model
#model_quartz = tcsconv(filters=128, kernel_size=39, input_dim=128, input_length=128)
model_quartz = quartz_model()
print(model_quartz.summary()) #show the structure of model

train_model(input_to_softmax=model_quartz,
            pickle_path='model_test.pickle',
            spectrogram=True)
