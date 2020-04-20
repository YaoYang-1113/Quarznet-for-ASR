# Quarznet-for-ASR
## This project is followed by >>[lucko515](https://github.com/lucko515/speech-recognition-neural-network)
### **How to start**
This project consists of two subfolders and 12 code files. Before using it, you need to prepare an proper environment and download dataset according to the Lucko script instruction. When everything is done, you could simply apply it for speech recognition or training a better model by your own.
### **Loss in results folder**
You can see that validation loss on both two models in trainning process are so strange while unfortunely I couldn't figure out the reason. On the other hand, these models performs not so bad cause WER(word error rate) on validation set is around 50%. At teh same time if you could make some changes on decoder such as >[**Token Passing**](https://github.com/githubharald/CTCDecoder) which I am doing now, then final results would be much better.
![Loss](https://github.com/mribles/Quarznet-for-ASR/blob/master/Loss.png)
