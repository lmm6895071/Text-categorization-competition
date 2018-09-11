#-*- coding: utf-8 -*-
from __future__ import unicode_literals
#author: Zhen Wu

import numpy as np
import os,sys,pickle

def load_embedding(embedding_file_path, corpus, embedding_dim):
    #wordset = set();
    # for line in corpus:
    #     line = line.strip().split()
    #     for w in line:
    #         w=w.strip()
    #         if len(w)<1:
    #             continue
    #         wordset.add(w.lower())
    # print wordset
    #wordset=set(corpus)
    word_embedding,words_dict=read_pretrained_word2vec(embedding_file_path, corpus, embedding_dim)
    return word_embedding,words_dict
    '''
    words_dict = dict(); word_embedding = []; index = 1
    words_dict['$EOF$'] = 0  #add EOF
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r') as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = [float(s) for s in line[1:]]
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return np.asarray(word_embedding), words_dict
    '''

def read_pretrained_word2vec( path, vocab, dim):
    isFirst=False

    if os.path.isfile(path):
        if isFirst ==True:
            raw_word2vec = open(path, 'r')
        else:
            with open( "data/save_wordsdict1.pickle", 'rb') as f:
                word2vec_dic = pickle.load(f)
            with open( "data/save_embedding.pickle", 'rb') as f:
                word_embedding = pickle.load(f)
            return np.asarray(word_embedding),word2vec_dic

    else:
        print "Path (word2vec) is wrong!"
        sys.exit()

    word2vec_dic = dict(); word_embedding = []; index = 1
    # word2vec_dic['$EOF$'] = 0  #add EOF
    word_embedding.append(np.zeros(dim))

    all_line = raw_word2vec.read().splitlines()
    mean = np.zeros(dim)
    count = 0
    for line in all_line:
        tmp = line.split()
        _word = tmp[0]
        _vec = np.array(tmp[1:], dtype=float)
        # print len(_vec)

        if _vec.shape[0] != dim:
            print "Mismatch the dimension of pre-trained word vector with word embedding dimension!"
            print index
            print _word
            # sys.exit()

        if _word not in vocab:
            # _vec=np.random.normal(mean, 0.1, size=dim)
            continue
        else:
            count = count + 1
        if index ==5000:
            break

        word_embedding.append(_vec)
        word2vec_dic[_word] = index
        index = index + 1


    print "%d words exist in the given pretrained model" % count


    with open(  "data/save_wordsdict1.pickle", 'wb') as f:
        pickle.dump(word2vec_dic, f)
    with open("data/save_embedding.pickle", 'wb') as f:
        pickle.dump(word_embedding, f)

    return np.asarray(word_embedding), word2vec_dic


def fit_transform(x_text, words_dict, max_sen_len, max_doc_len):
    x, sen_len, doc_len = [], [], []
    for index, doc in enumerate(x_text):
        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=int)
        sentences = doc.split('\n')
        i = 0
        for sen in sentences:
            j = 0
            for word in sen.strip().split():
                if j >= max_sen_len:
                    break
                if word not in words_dict: continue
                t_x[i, j] = words_dict[word]
                j += 1
            t_sen_len[i] = j
            i += 1
            if i >= max_doc_len:
                break
        doc_len.append(i)
        sen_len.append(t_sen_len)
        x.append(t_x)
    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len)

class Dataset(object):
    def __init__(self, data_file,final_test=False):
        self.t_label = []
        self.t_docs = []
        self.t_ids=[]

        test_count = 0
        with open(data_file, 'r') as f:
            for line in f:
                # line = line.strip().decode('utf8', 'ignore').split('\t\t')
                lines = line.strip().split('::')
                if len(lines)<2:
                    print "test", line,lines
                    continue
                if final_test ==False:
                    self.t_label.append(int(lines[1])-1)
                    self.t_docs.append(lines[0])
                else:
                    self.t_label.append(0)
                    self.t_docs.append(lines[0])
                    self.t_ids.append(lines[1])
                    test_count=test_count+1

        self.data_size = len(self.t_docs)
    def genBatch(self,wordsdict, batch_size, max_sen_len, max_doc_len, n_class,final_test=False):
        self.epoch = len(self.t_docs) / batch_size
        if len(self.t_docs) % batch_size != 0:
            self.epoch += 1
        self.label = []
        self.docs = []
        self.sen_len = []
        self.doc_len = []
        self.test_id=[]

        for i in xrange(self.epoch):
            if final_test == True:
                self.test_id.append(self.t_ids[i*batch_size:(i+1)*batch_size])
            
            self.label.append(np.eye(n_class, dtype=np.float32)[self.t_label[i*batch_size:(i+1)*batch_size]])

            b_docs, b_sen_len, b_doc_len = fit_transform(self.t_docs[i*batch_size:(i+1)*batch_size],
                                                         wordsdict, max_sen_len, max_doc_len)
            self.docs.append(b_docs)
            self.sen_len.append(b_sen_len)
            self.doc_len.append(b_doc_len)

    def batch_iter(self, counts,wordsdict, n_class, batch_size, num_epochs, max_sen_len, max_doc_len, shuffle=True):
        data_size =10000# len(self.t_docs)
        
        sstart= data_size*counts
        send = data_size*(counts+1)
        if len(self.t_label) <send:
            send=len(self.t_label)
            data_size=send-sstart
        t_label = np.asarray(self.t_label[sstart:send])
        t_docs = np.asarray(self.t_docs[sstart:send])

        num_batches_per_epoch = int(data_size / batch_size) + \
                                (1 if data_size % batch_size else 0)

        for epoch in xrange(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                t_label = t_label[shuffle_indices]
                t_docs = t_docs[shuffle_indices]

            for batch_num in xrange(num_batches_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                label = np.eye(n_class, dtype=np.float32)[t_label[start:end]]

                docs, sen_len, doc_len = fit_transform(t_docs[start:end], wordsdict, max_sen_len, max_doc_len)
                batch_data = zip(docs, label, sen_len, doc_len)
                yield batch_data