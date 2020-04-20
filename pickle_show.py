import pickle
f = open('results/model_LSTM_1.pickle','rb')
info = pickle.load(f)
print (info)   #show file