from test_utils import *
from quartznet import *
#plot_raw_audio(index=100)
#plot_mfcc_feature(index=0)
#plot_spectrogram_feature(index=1104)
model_quartz = quartz_same_model()
lexcion_predictions(index=495,
                partition='validation',
                input_to_softmax=model_quartz,
                model_path='results/model_quartznet_same_epoch-40.hdf5')
get_predictions(index=495,                  # 1424,1044,2458,1299,122,412                   1210,1460,553,2368,1912,
                partition='validation',        # change to train or validation
                input_to_softmax=model_quartz,
                model_path='results/model_quartznet.hdf5')
'''get_group_predictions(input_to_softmax=model_quartz,
                      model_path='results/model_quartznet.hdf5',
                      partition='validation')
validation_sentences()'''