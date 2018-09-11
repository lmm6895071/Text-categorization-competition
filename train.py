#-*- coding: utf-8 -*-
from __future__ import unicode_literals
#author: Ming Li

import os, time, pickle
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
from model_classification import model_classification
import datetime
from sklearn import metrics
from sklearn.metrics import classification_report


# Data loading params
tf.flags.DEFINE_integer("n_class", 19, "Numbers of class")
tf.flags.DEFINE_string("dataset", '', "The dataset")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("hidden_size", 100, "hidden_size of rnn")
tf.flags.DEFINE_integer('max_sen_len', 400, 'max number of tokens per sentence')
tf.flags.DEFINE_integer('max_doc_len', 1, 'max number of tokens per sentence')
tf.flags.DEFINE_float("lr", 0.001, "Learning rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("model_type","classification","model type classification or regression")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Load data
print("Loading data...")

stime = time.time()
trainset = Dataset('data/train.txt')
etime = time.time()
print "================= load trainset ===============",etime-stime
devset = Dataset('data/dev.txt')
stime = time.time()
print "================= load devset ===============",stime-etime
testset = Dataset('data/test.txt')
etime = time.time()
print "================= load testset ===============",etime-stime

# alldata = np.concatenate([trainset.t_docs, devset.t_docs, testset.t_docs], axis=0)

fs =open('data/wordlist.txt')
alldata=fs.readlines()
alldata = [item.strip() for item in alldata]
fs.close()
estime = time.time()
print "================= load wordsdict ===============",estime-etime

embeddingpath = 'data/embeding1'
embeddingfile, wordsdict = data_helpers.load_embedding(embeddingpath, alldata, FLAGS.embedding_dim)

print type(embeddingfile)
del alldata
stime=time.time()
print "================= load word2vec ===============",stime-estime

print("Loading data finished...")


devset.genBatch(wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)
etime= time.time()
print "================= devset genBatch ===============", etime-stime

testset.genBatch(wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)
stime = time.time()
print "================= testset genBatch ===============",stime-etime

print "satrt*********************************************************"

with tf.Graph().as_default():
    # session_config = tf.ConfigProto(
    #     allow_soft_placement=FLAGS.allow_soft_placement,
    #     log_device_placement=FLAGS.log_device_placement
    # )
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #session_config = tf.ConfigProto()
    #session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
    #session_config.gpu_options.allow_growth = True

    sess = tf.Session()#(config=session_config)
    with sess.as_default():

        if FLAGS.model_type == 'classification':
            huapa = model_classification(
                max_sen_len = FLAGS.max_sen_len,
                max_doc_len = FLAGS.max_doc_len,
                class_num = FLAGS.n_class,
                embedding_file = embeddingfile,
                embedding_dim = FLAGS.embedding_dim,
                hidden_size = FLAGS.hidden_size,
            )
        elif FLAGS.model_type == "regression":
            huapa = model_regression(
                max_sen_len = FLAGS.max_sen_len,
                max_doc_len = FLAGS.max_doc_len,
                class_num = FLAGS.n_class,
                embedding_file = embeddingfile,
                embedding_dim = FLAGS.embedding_dim,
                hidden_size = FLAGS.hidden_size,
            )
        huapa.build_model()
        print "build model*********************************************************"
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        grads_and_vars = optimizer.compute_gradients(huapa.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Save dict
        timestamp = str(int(time.time()))
        print timestamp
        checkpoint_dir = os.path.abspath("checkpoints/"+str(FLAGS.max_sen_len)+"/")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        # with open(checkpoint_dir + "/wordsdict1.txt", 'wb') as f:
        #     pickle.dump(wordsdict, f)

        sess.run(tf.global_variables_initializer())
        def train_step(batch):
            x,y,sen_len, doc_len = zip(*batch)
            feed_dict = {
                huapa.input_x: x,
                huapa.input_y: y,
                huapa.sen_len: sen_len,
                huapa.doc_len: doc_len
            }
            _, step, loss, y_pred,y_true = sess.run(
                [train_op, global_step, huapa.loss, huapa.y_pred,huapa.y_true],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            # print("{}".format( metrics.classification_report(y_true,y_pred)))
            print metrics.classification_report(y_true,y_pred).split("\n")[-2]
            # print metrics.confusion_matrix(y_true,y_pred)



        def predict_step( x, y, sen_len, doc_len, name=None):
            feed_dict = {
                huapa.input_x: x,
                huapa.input_y: y,
                huapa.sen_len: sen_len,
                huapa.doc_len: doc_len
            }
            step, loss, y_pred,y_true= sess.run(
                [global_step, huapa.loss, huapa.y_pred,huapa.y_true],
                feed_dict)
            return  y_pred,y_true

        def predict(dataset, name=None):
            all_y_true=[]
            all_y_pred=[]
            for i in xrange(dataset.epoch):
                y_pred,y_true= predict_step(dataset.docs[i],
                                                 dataset.label[i],dataset.sen_len[i], dataset.doc_len[i], name)
                all_y_true =all_y_true+ y_true.tolist()
                all_y_pred = all_y_pred+y_pred.tolist()
            # print metrics.f1_score(all_y_true,all_y_pred)
            # print metrics.classification_report(all_y_true,all_y_pred)
            print "==================",metrics.classification_report(all_y_true,all_y_pred).split("\n")[-2]
            # print metrics.confusion_matrix(all_y_true,all_y_pred)
            #return acc, rmse,smae
            return float(metrics.classification_report(all_y_true,all_y_pred).split("\n")[-2].split("    ")[-3])


        topf1 = 0.
        toprmse = 0.
        better_dev_acc = 0.
        better_dev_rmse =100
        predict_round = 0
        test_f1=0.0

        for ii in range(FLAGS.num_epochs):

            sum_len= len(trainset.t_label)
            counts= sum_len/10000+1
            global_steps=0
            for count in range(counts):
                trainbatches = trainset.batch_iter(count,wordsdict, FLAGS.n_class, FLAGS.batch_size,
                                     1, FLAGS.max_sen_len, FLAGS.max_doc_len)
                # Training loop. For each batch...
                for tr_batch in trainbatches:
                    train_step(tr_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0 or ii==FLAGS.num_epochs:
                        predict_round += 1
                        print("\nEvaluation round %d:" % (predict_round))

                        print "==========devset===========" 
                        predict(devset, name="dev")
                        #print("dev_acc: %.4f    dev_RMSE: %.4f   dev_MAE: %.4f" % (dev_acc, dev_rmse,dev_mae))
                        test_f1=predict(testset, name="test")
                        #print("test_acc: %.4f    test_RMSE: %.4f   test_MAE: %.4f" % (test_acc, test_rmse,test_mae)
                        global_steps = current_step

                        if test_f1 >=topf1:
                            topf1 = test_f1
                            path = saver.save(sess, checkpoint_prefix)
                            print("Saved model checkpoint to {}\n".format(current_step))
                        print "=================final result is================= ",topf1

        print "final result is ",topf1
