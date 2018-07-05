import tensorflow as tf
import numpy as np
import datetime
import time
import os
import data_helper2
from FRDNN_MODEL import Model

total_s = 0

print("...Load data...")
data, label = data_helper2.load_data_and_labels("./data/2330.TW.csv")  #size:4499/4480

# Split data and label set
split_data_index = (data_helper2.conf.train_batch+data_helper2.conf.data_batch_size-2)
split_label_index = (data_helper2.conf.train_batch+data_helper2.conf.label_batch_size-2)
training_data, training_label = data[:split_data_index+1], label[:split_label_index+1]
testing_data, testing_label = data[data_helper2.conf.train_batch-1:], label[data_helper2.conf.train_batch-1:4469]

def Training_Data_Batch(i):
    return training_data[i:i + data_helper2.conf.data_batch_size]


def Training_Label_Batch(i):
    return training_label[i:i + data_helper2.conf.label_batch_size]


def Testing_Data_Batch(i):
    return testing_data[i:i + data_helper2.conf.data_batch_size]


def Testing_Label_Batch(i):
    return testing_label[i:i + data_helper2.conf.label_batch_size]

# Training
# ==================================================

#Making a big flow chart tf.Graph
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)

    # Making a small flow chart tf.sess
    with sess.as_default():
        model = Model(total_data=data)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=data_helper2.conf.learning_rate)
        train_op = optimizer.minimize(model.loss, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", model.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test summaries
        dev_summary_op = tf.summary.merge([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # sess end
        # Initialize all variables
        sess.run(tf.global_variables_initializer())  # when it's end,initialize all variables

        def train_step(data_batch, label_batch):
            feed_dict = {
                model.fuzzyIn: data_batch,
                model.labelIn: label_batch}
            _, step, loss, result = sess.run(
                [train_op, global_step, model.loss, model.outputs],
                feed_dict)
            train_accuracy = sess.run(model.loss, feed_dict={
                model.fuzzyIn: data_batch, model.labelIn: label_batch, })
            time_str = datetime.datetime.now().isoformat()

            data_helper2.conf.p_s = 0
            data_helper2.conf.p_s += prediction(result, label_batch)

            a = float((data_helper2.conf.p_s /total_s) * 100)

            print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, a))

        def test_step(data_batch, label_batch, writer=None):
            # a test step
            feed_dict = {
                model.fuzzyIn: data_batch,
                model.labelIn: label_batch}
            step, summaries, loss = sess.run(
                [global_step, dev_summary_op, model.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            test_accuracy = sess.run(model.loss, feed_dict={
                model.fuzzyIn: data_batch, model.labelIn: label_batch})
            print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, test_accuracy))
            if writer:
                writer.add_summary(summaries, step)

        def prediction(result, label):
            List = []
            List2 = []
            match_count = 0
            for l in range(data_helper2.conf.label_batch_size):
                if result[l] < -0.5:
                    p = -1
                    List.append(p)
                elif -0.5 <= result[l] <= 0.5:
                    p = 0
                    List.append(p)
                else:
                    p = 1
                    List.append(p)
            for q in range(len(List)):
                if List[q] == label[q]:
                    List2.append(1)
                    print(1)
                else:
                    List2.append(0)
                    print(0)
            for w in range(len(List2)):
                match_count += List2[w]
            return match_count
        # Training loop. For each batch...
        for epoch in range(data_helper2.conf.num_epochs):
            for j in range(data_helper2.conf.train_batch):
                total_s += 20
                x_batch, y_batch = Training_Data_Batch(j), Training_Label_Batch(j)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                #print(current_step)

        for epoch2 in range(data_helper2.conf.num_epochs):
            for k in range(data_helper2.conf.test_batch):
                x_batch, y_batch = Testing_Data_Batch(k), Testing_Label_Batch(k)
                test_step(x_batch, y_batch)
                #print(current_step)

            if current_step % 100 == 0:
                print("\nEvaluation:")
                test_step(testing_data[:50], testing_label[:20], writer=dev_summary_writer)
                print("")
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))