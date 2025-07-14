import numpy as np
import pandas as pd
import wave
import jieba
import re
# from allennlp.commands.elmo import ElmoEmbedder
import os
import csv
import pandas as pd
prefix = os.path.abspath(os.path.join(os.getcwd(), "."))
answers = []
train_targets = {}
dev_targets = {}
test_targets = {}
def collect_data( path):
    ### train label
    train_=pd.read_csv(prefix+path+'/train_split_Depression_AVEC2017.csv')
    dev_ = pd.read_csv(prefix + path + '/dev_split_Depression_AVEC2017.csv')
    test_ = pd.read_csv(prefix + path + '/test_split_Depression_AVEC2017.csv')
    for i in range(1,train_.shape[0]):
        train_targets[train_.loc[i,'Participant_ID']]=train_.loc[i,'PHQ8_Binary']
    with open('./train_label_DAIC.txt', 'w') as fw:
        fw.write(str(train_targets))
        fw.close()
    ### eval  label
    for i in range(1,  dev_.shape[0]):
        dev_targets[ dev_.loc[i,'Participant_ID']] =  dev_.loc[i,'PHQ8_Binary']
    with open('./dev_label_DAIC.txt', 'w') as fw:
        fw.write(str(dev_targets))
        fw.close()


if __name__ == '__main__':
    path='/Dataset/DAIC-WOZ'
    collect_data(path)