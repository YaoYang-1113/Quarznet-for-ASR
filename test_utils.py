import random
import numpy as np
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
from IPython.display import Audio
from jiwer import wer
import time
import tensorflow as tf
from main import testline


def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]


def get_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions ·
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    starttime = time.time()
    data_gen = AudioGenerator(spectrogram=True)
    data_gen.load_train_data()
    data_gen.load_validation_data()
    # obtain the true transcription and the audio features
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    #print(input_to_softmax.summary())
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    output_length = [input_to_softmax.output_length(data_point.shape[0])]
    pred_ints = (K.eval(K.ctc_decode(
       prediction, output_length, greedy=True)[0][0]) + 1).flatten().tolist()
    Audio(audio_path)
    print('-' * 80)
    b = "".join(int_sequence_to_text(pred_ints))
    a = transcr
    print("Greedy_predictions:\n" + '\n' + ''.join(int_sequence_to_text(pred_ints)))
    print('1. Word Error Rate for ASR ==', wer(a, b)*100, '%')
    endtime = time.time()
    print('2. Real Time Factor for ASR ==',(endtime - starttime)/data_gen.valid_durations[index],'\n')
    print('-' * 80)

def lexcion_predictions(index, partition, input_to_softmax, model_path):
    """ Print a model's decoded predictions ·
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        input_to_softmax (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    starttime = time.time()
    data_gen = AudioGenerator(spectrogram=True)
    data_gen.load_train_data()
    data_gen.load_validation_data()
    # obtain the true transcription and the audio features
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    # obtain and decode the acoustic model's predictions
    input_to_softmax.load_weights(model_path)
    #print(input_to_softmax.summary())
    prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
    print('-' * 80)
    Audio(audio_path)
    print('True transcription:\n' + '\n' + transcr)
    print('-' * 80)
    b = testline(prediction[0])
    a = transcr
    print("TokenPassing_predictions:\n")
    print(b + '\n')
    print('1. Word Error Rate for ASR ==', wer(a, b)*100, '%')
    endtime = time.time()
    print('2. Real Time Factor for ASR ==',(endtime - starttime)/data_gen.valid_durations[index])

def validation_sentences():
    """ storage the validation sentences
    Params:
        None
    """
    # load the test data
    data_gen = AudioGenerator(spectrogram=True)
    data_gen.load_validation_data()
    # obtain the true transcription and the audio features
    num = 500
    f = open('C:/Users/mribles/Desktop/corpus.txt', 'a')
    while num > 490:
        transcr = data_gen.valid_texts[num]
        f.write(transcr + '\n')
        num = num -1
    f.close()

def get_group_predictions(input_to_softmax, model_path, partition):
    starttime = time.time()
    wer_sum = 0
    data_gen = AudioGenerator(spectrogram=True)
    data_gen.load_train_data()
    data_gen.load_validation_data()
    input_to_softmax.load_weights(model_path)
    # obtain the true transcription and the audio features
    if partition == 'validation':
        num = 99
        while num >= 0:
            index = random.randint(1, 2500)
            transcr = data_gen.valid_texts[index]
            audio_path = data_gen.valid_audio_paths[index]
            data_point = data_gen.normalize(data_gen.featurize(audio_path))
            # obtain and decode the acoustic model's predictions
            prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
            output_length = [input_to_softmax.output_length(data_point.shape[0])]
            pred_ints = (K.eval(K.ctc_decode(prediction, output_length, greedy=False, beam_width=100, top_paths=1)[0][
                                    0]) + 1).flatten().tolist()
            # print('True transcription:\n' + '\n' + transcr)
            b = "".join(int_sequence_to_text(pred_ints))
            a = transcr
            # print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
            # print('-' * 80)
            # print('1.Editable Distance for ASR ==', edit(a, b), '\n')
            if (wer(a, b) <= 1):
                print('index_%d' % index, ':')
                wer_sum = wer_sum + wer(a, b)
                print(wer(a, b))
                print("Transcription: ",a)
                print("Prediction:    ",b)
                print('-' * 80)
            elif ():
                num = num + 1
            num = num - 1
    elif partition == 'train':
        num = 999
        while num >= 0:
            index = random.randint(1, 10000)
            transcr = data_gen.train_texts[index]
            audio_path = data_gen.train_audio_paths[index]
            data_point = data_gen.normalize(data_gen.featurize(audio_path))

            # obtain and decode the acoustic model's predictions

            prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
            output_length = [input_to_softmax.output_length(data_point.shape[0])]
            pred_ints = (K.eval(K.ctc_decode(prediction, output_length, greedy=False, beam_width=100, top_paths=1)[0][
                                    0]) + 1).flatten().tolist()
            # print('True transcription:\n' + '\n' + transcr)
            b = "".join(int_sequence_to_text(pred_ints))
            a = transcr
            # print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
            # print('-' * 80)
            # print('1.Editable Distance for ASR ==', edit(a, b), '\n')

            if (wer(a, b) <= 1):
                print('index_%d' % index, ':')
                wer_sum = wer_sum + wer(a, b)
                print(wer(a, b))
                print("Transcription: ",a)
                print("Prediction:    ",b)
                print('-' * 80)
            elif ():
                num = num + 1
            num = num - 1
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    endtime = time.time()
    #print('1. Average Word Error Rate for ASR ==', wer_sum/100 , '%')
    print('1. Average Word Error Rate for ASR ==', wer_sum / 100)
    print('2. Average Real Time Factor for ASR ==', (endtime - starttime) / 100, '\n')
