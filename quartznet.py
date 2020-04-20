from keras.models import Model
from keras.layers import (BatchNormalization, Input, Activation, Add)
from keras.layers.convolutional import SeparableConv1D

def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=2):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def ol(input_length):
    output_length = input_length - 204
    return output_length


def C(filters, kernel_size, input_dim, conv_border_mode, conv_stride,dilation):              # Layer C1: first layer
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv = SeparableConv1D(filters=filters, kernel_size=kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     dilation_rate=dilation,
                     depth_multiplier=1,
                     activation=None,
                     use_bias=True,
                     depthwise_initializer='glorot_uniform',
                     pointwise_initializer='glorot_uniform',
                     bias_initializer='zeros',
                     name='c')(input_data)
    conv_bn = BatchNormalization(name='batch_norm1')(conv)
    relu = Activation('relu', name='relu1')(conv_bn)
    # Specify the model
    model = Model(inputs=input_data, outputs=relu)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length(x,kernel_size,conv_border_mode,conv_stride,dilation)
    return model


def tcsconv(filters, kernel_size, input_dim):         # Layer B: output_length = input_length
    input_data = Input(shape=(None, input_dim))
    x0 = SeparableConv1D(filters, kernel_size=1,  # x0: pointwise + batchnorm
                strides=1,
                padding='same',
                activation=None,
                name='c1')(input_data)
    x0 = BatchNormalization(name='batch_norm1')(x0)
    # x1*1
    x1 = SeparableConv1D(filters, kernel_size,  # x1: normal separable conv + batchnorm + relu
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         dilation_rate=1,
                         depth_multiplier=1,
                         activation=None,
                         use_bias=True,
                         depthwise_initializer='glorot_uniform',
                         pointwise_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name='layer_separable_conv1')(input_data)
    x1 = BatchNormalization(name='conv_batch_norm1')(x1)
    x1 = Activation('relu', name='relu1')(x1)
    # x1*2
    x1 = SeparableConv1D(filters, kernel_size,  # x1: normal separable conv + batchnorm + relu
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         dilation_rate=1,
                         depth_multiplier=1,
                         activation=None,
                         use_bias=True,
                         depthwise_initializer='glorot_uniform',
                         pointwise_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name='layer_separable_conv2')(x1)
    x1 = BatchNormalization(name='conv_batch_norm2')(x1)
    x1 = Activation('relu', name='relu2')(x1)
    # x1*3
    x1 = SeparableConv1D(filters, kernel_size,  # x1: normal separable conv + batchnorm + relu
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         dilation_rate=1,
                         depth_multiplier=1,
                         activation=None,
                         use_bias=True,
                         depthwise_initializer='glorot_uniform',
                         pointwise_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name='layer_separable_conv3')(x1)
    x1 = BatchNormalization(name='conv_batch_norm3')(x1)
    x1 = Activation('relu', name='relu3')(x1)
    # x1*4
    x1 = SeparableConv1D(filters, kernel_size,  # x1: normal separable conv + batchnorm + relu
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         dilation_rate=1,
                         depth_multiplier=1,
                         activation=None,
                         use_bias=True,
                         depthwise_initializer='glorot_uniform',
                         pointwise_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name='layer_separable_conv4')(x1)
    x1 = BatchNormalization(name='conv_batch_norm4')(x1)
    x1 = Activation('relu', name='relu4')(x1)
    # x2*1
    x2 = SeparableConv1D(filters, kernel_size,  # x2: normal separable conv + batchnorm
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         dilation_rate=1,
                         depth_multiplier=1,
                         activation=None,
                         use_bias=True,
                         depthwise_initializer='glorot_uniform',
                         pointwise_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         name='layer_separable_conv')(x1)
    x2 = BatchNormalization(name='conv_batch_norm')(x2)

    added = Add()([x0, x2])
    out = Activation('relu', name='relu')(added)  # out: out(x3) := ( x1 ->x1 ->x1 ->x1 ->x2) + x0
    tcsconv_B = Model(inputs=input_data, outputs=out)
    #tcsconv_B.output_length = lambda x: input_length
    return tcsconv_B

def Pointwise(filters, kernel_size, input_dim, conv_border_mode, conv_stride,dilation):
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv = SeparableConv1D(filters=filters, kernel_size=kernel_size,
                     strides=1,
                     padding=conv_border_mode,
                     name='c')(input_data)
    # Specify the model
    model = Model(inputs=input_data, outputs=conv)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length(x,kernel_size,conv_border_mode,conv_stride,dilation)
    return model

def quartz_valid_model():
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, 161))
    conv1 = C(filters=128,kernel_size=33,input_dim=161,conv_border_mode='same',conv_stride=1,dilation=1)(input_data)      #output_length = input_length

    B1 = tcsconv(filters=128, kernel_size=33, input_dim=128)(conv1) #output_length = input_length
    B2 = tcsconv(filters=128, kernel_size=39, input_dim=128)(B1)    #output_length = input_length
    B3 = tcsconv(filters=128, kernel_size=51, input_dim=128)(B2)    #output_length = input_length
    B4 = tcsconv(filters=128, kernel_size=63, input_dim=128)(B3)    #output_length = input_length
    B5 = tcsconv(filters=128, kernel_size=75, input_dim=128)(B4)    #output_length = input_length

    conv2 =  C(filters=256,kernel_size=87,input_dim=128,conv_border_mode='valid',conv_stride=1,dilation=2)(B5)             #output_length != input_length
    conv3 =  C(filters=512, kernel_size=1, input_dim=256, conv_border_mode='same', conv_stride=1,dilation=1)(conv2)       #output_length = input_length
    conv4 = Pointwise(filters=29, kernel_size=1, input_dim=512, conv_border_mode='same', conv_stride=1,dilation=1)(conv3) #output_length = input_length

    y_pred = Activation('softmax', name='softmax')(conv4)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, 87, 'valid', 1, 2)
    return model

def quartz_same_model():
    """ Build a deep network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, 161))
    conv1 = C(filters=128,kernel_size=33,input_dim=161,conv_border_mode='same',conv_stride=1,dilation=1)(input_data)      #output_length = input_length

    B1 = tcsconv(filters=128, kernel_size=33, input_dim=128)(conv1) #output_length = input_length
    B2 = tcsconv(filters=128, kernel_size=39, input_dim=128)(B1)    #output_length = input_length
    B3 = tcsconv(filters=128, kernel_size=51, input_dim=128)(B2)    #output_length = input_length
    B4 = tcsconv(filters=128, kernel_size=63, input_dim=128)(B3)    #output_length = input_length
    B5 = tcsconv(filters=128, kernel_size=75, input_dim=128)(B4)    #output_length = input_length

    conv2 =  C(filters=256,kernel_size=87,input_dim=128,conv_border_mode='same',conv_stride=1,dilation=2)(B5)             #output_length != input_length
    conv3 =  C(filters=512, kernel_size=1, input_dim=256, conv_border_mode='same', conv_stride=1,dilation=1)(conv2)       #output_length = input_length
    conv4 = Pointwise(filters=29, kernel_size=1, input_dim=512, conv_border_mode='same', conv_stride=1,dilation=1)(conv3) #output_length = input_length

    y_pred = Activation('softmax', name='softmax')(conv4)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, 87, 'same', 1, 2)
    return model