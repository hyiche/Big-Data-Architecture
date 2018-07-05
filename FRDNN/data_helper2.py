import collections
import pandas as pd
import numpy as np

conf = collections.namedtuple("conf","data_batch_size,label_batch_size,num_epochs,train_batch,test_batch,"
                                     "init_step_learning_rate,learning_rate")
conf.data_batch_size = 50
conf.label_batch_size = 20
conf.num_epochs = 1
conf.train_batch = 4000
conf.test_batch = 450
conf.init_step_learning_rate = 0.001
conf.learning_rate = 0.001

def load_data_and_labels(FilePath):
    df = pd.read_csv(FilePath, header=None, index_col=None)
    df = df.loc[1:, 4]
    df = np.asarray(a=df, dtype=np.float32)
    df = df.reshape(-1, 1)
    data = []
    for i in range(1, len(df)):
        data.append(df[i][0] - df[i - 1][0])  # generate spread data
    data = np.asarray(a=data, dtype=np.float32)
    data = data.reshape(-1, 1)
    # generate labels
    for j in range(len(df) - conf.label_batch_size):  # df as label
        if df[j + conf.label_batch_size][0] > df[j][0]:
            df[j][0] = 1
        elif df[j + conf.label_batch_size][0] == df[j][0]:
            df[j][0] = 0
        else:
            df[j][0] = -1
    df = df[:len(df) - conf.label_batch_size]
    return [data, df]

