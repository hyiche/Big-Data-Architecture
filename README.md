# Big-Data-Architecture
#碩士班一年級期末小組報告
#import
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#input data(close price) and translate to value
tsmc_stocks = pd.read_csv("D://data/2353.TW.csv")
data_to_use = tsmc_stocks['Close'].values

scaled_dataset = data_to_use.reshape(-1,1)

'''
#scaling data
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(data_to_use.reshape(-1,1))
'''

#make stock price into lag return list
y_label = []
i = 0
while (i) <= len(scaled_dataset)-1:
    if i >= 20:
        a = scaled_dataset[i] - scaled_dataset[i-1]
        y_label.append(a)
    else:
        y_label.append(0)
    i += 1
y_label = np.array(y_label)
y_label = y_label.reshape(-1,1)


#Windowing the dataset
def window_data(y_label,window_size):
    X = []
    y = []

    i = 0
    while (i + window_size) <= len(y_label) - 1:
        X.append(y_label[i:i + window_size])
        y.append(y_label[i + window_size])

        i += 1
    assert len(X) == len(y)
    return X, y

X, y = window_data(y_label, 20)


#Creating Training and Testing sets
X_train  = np.array(X[:4200])
X_test = np.array(X[4200:])
y_train = np.array(y[:4200])
y_test = np.array(y[4200:])




#create the NN+LSTM

def fully_connected(input):

    layer_1 = tf.contrib.layers.fully_connected(input,10,activation_fn = tf.nn.relu)
    layer_1_drop = tf.contrib.layers.dropout(layer_1,keep_prob =0.5)

    layer_2 = tf.contrib.layers.fully_connected(layer_1_drop,10,activation_fn =tf.nn.relu)
    layer_2_drop = tf.contrib.layers.dropout(layer_2,keep_prob = 0.5)

    return layer_2_drop

def LSTM_cell(hidden_layer_size, batch_size,number_of_layers, dropout, dropout_rate):

    layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)

    if dropout:
        layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)

    cell = tf.contrib.rnn.MultiRNNCell([layer]*1)

    init_state = cell.zero_state(batch_size, tf.float32)

    return cell, init_state

def output_layer(lstm_output, in_size, out_size):

    x = lstm_output[:, -1, :]
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05))
    bias = tf.Variable(tf.zeros([out_size]))

    output = tf.matmul(x, weights) + bias
    output = tf.tanh(output)

    return output

#Hyper parameter
window_size=20
number_of_classes=1
learning_rate=0.001
batch_size=1
hidden_layer_size=15
number_of_layers=1
dropout=True
dropout_rate=0.5
epochs = 20
cost = 0.0

#palceholder
input_stocks = tf.placeholder(tf.float32, [batch_size, window_size, 1], name='input_data')
return_label = tf.placeholder(tf.float32, [batch_size, 1], name='targets')

#go through the network
output_fully = fully_connected(input_stocks)
cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout, dropout_rate)
outputs_LSTM, states = tf.nn.dynamic_rnn(cell, output_fully, initial_state=init_state)
final_output = output_layer(outputs_LSTM,hidden_layer_size, number_of_classes)


#loss ( total return )
final_output_1 = 0
pre_return = 0
Return = final_output*return_label-cost*np.absolute(final_output-final_output_1)
Return = pre_return+Return

#train op
optimizer = tf.train.AdamOptimizer(learning_rate)
train_optimizer = optimizer.minimize(-1*Return)

#training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epochs):
    ii = 0
    epoch_loss = []
    final_output_1 = 0
    pre_return = 0
    while (ii + batch_size) <= len(X_train):
        X_batch = X_train[ii:ii + batch_size]
        y_batch = y_train[ii:ii + batch_size]

        o, c, _ = sess.run([final_output,Return,train_optimizer],
                            feed_dict={input_stocks: X_batch, return_label: y_batch})
        pre_return = c
        final_output_1 = o

        epoch_loss.append(c)
        ii += batch_size
    if (i % 1) == 0:
        print('Epoch {}/{}'.format(i, epochs), ' Current return: {}'.format(np.mean(epoch_loss)))

    print(o)

#plot the result

tests = []
i = 0
while i+batch_size  <= len(X_test):
    o = sess.run([final_output], feed_dict={input_stocks: X_test[i:i + batch_size]})
    i += batch_size
    tests.append(o)

tests = np.array(tests)

decsion = []
for i in range(len(tests)):
    if tests[i] > 0.5:
        decsion.append(1)
    elif tests[i] < -0.5:
        decsion.append(-1)
    else:
        decsion.append(0)


decsion = np.array(decsion)
y_test = y_test.reshape(1,-1)
test_return = decsion*y_test


result_return = []
for i in range(test_return.shape[1]):
    result_return.append(test_return[0][i])

avaerge_return = np.mean(result_return)
print(avaerge_return)


plt.figure(figsize=(16, 7))
plt.title('acer*s trading return from MODRL')
plt.xlabel('Days')
plt.ylabel('Return')
plt.plot(result_return)
plt.show()
