#-*- coding: utf-8 -*-
from __future__ import unicode_literals
#author: aaronzark

import datetime, os, time, pickle
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
import pandas as pd


# Data loading params
tf.flags.DEFINE_integer("n_class", 19, "Numbers of class")
tf.flags.DEFINE_string("dataset", '', "The dataset")
tf.flags.DEFINE_integer('max_sen_len', 400, 'max number of tokens per sentence')
tf.flags.DEFINE_integer('max_doc_len', 1, 'max number of tokens per sentence')
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
 

# Load data
checkpoint_dir = os.path.abspath("checkpoints/"+str(FLAGS.max_sen_len)+"/")
checkpoint_prefix = os.path.join("./checkpoints/"+str(FLAGS.max_sen_len)+"/", "model")
# checkpoint_file = tf.train.import_meta_graph(checkpoint_prefix+".meta")
print "====",checkpoint_prefix


stime = time.time()
testset = Dataset('data/final_test.txt',True)
etime = time.time()
print "================= load testset ===============",etime-stime
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


testset.genBatch(wordsdict, FLAGS.batch_size, FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class,True)
sstime = time.time()
print "================= testset genBatch ===============",sstime-stime

print "satrt*********************************************************"

 
print("Loading data finished...")


graph = tf.Graph()
with graph.as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_prefix))
        saver.restore(sess, checkpoint_prefix)

        huapa_input_x = graph.get_operation_by_name("input/input_x").outputs[0]
        huapa_input_y = graph.get_operation_by_name("input/input_y").outputs[0]
        huapa_sen_len = graph.get_operation_by_name("input/sen_len").outputs[0]
        huapa_doc_len = graph.get_operation_by_name("input/doc_len").outputs[0]

      
        huapa_y_pred = graph.get_operation_by_name("softmax/predictions").outputs[0]
        huapa_y_proba = graph.get_operation_by_name("softmax/score").outputs[0]

        def predict_step( x, y, sen_len, doc_len, name=None):
            feed_dict = {
                huapa_input_x: x,
                huapa_input_y: y,
                huapa_sen_len: sen_len,
                huapa_doc_len: doc_len
            }
            y_pred,proba= sess.run([huapa_y_pred,huapa_y_proba],feed_dict)

            
            return  y_pred,proba

        def predict(dataset, name=None):
            all_y_id=[]
            all_y_pred=[]
            all_proba=[]
            print "=================================="
            for i in xrange(dataset.epoch):
                y_pred,max_proba= predict_step(dataset.docs[i],dataset.label[i],dataset.sen_len[i], dataset.doc_len[i], name)
                all_y_id =all_y_id+ dataset.test_id[i]
                all_y_pred = all_y_pred+list(y_pred.tolist())
                all_proba = all_proba+max_proba.tolist()
            
            return all_y_pred,all_y_id,np.array(all_proba)

        pred, test_id,all_proba = predict(testset, name="test")

        print "=======\n",len(test_id),len(pred)
        print "======="
        test_pred=pd.DataFrame(pred)                                       
        test_pred.columns=["class"]                                              
        test_pred["class"]=(test_pred["class"]+1).astype(int)                    
        print(test_pred.shape)                                                   
        print len(test_id)                                                       
        test_pred["id"]=test_id                                                  
        test_pred[["id","class"]].to_csv('data/attention_score1.csv',index=None)
 
      

        test_p1=pd.DataFrame(all_proba)
        test_p1.columns=["probability_%s"%i for i in range(1,all_proba.shape[1]+1)]
        print(test_p1.shape)
        print len(test_id)
        test_p1["id"]=test_id
        test_p1.to_csv('data/attention_probability1.csv',index=None)



