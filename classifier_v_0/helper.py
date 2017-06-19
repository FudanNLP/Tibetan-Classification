import numpy as np
import operator
from collections import defaultdict
import logging
from TfUtils import *
from numpy import indices

class Vocab(object):
    def __init__(self, unk='<unk>'):
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.unknown = unk
        self.add_word(self.unknown, count=0)

        
    def add_word(self, word, count=1):
        word = word.strip()
        if len(word) == 0:
            return
        elif word.isspace():
            return
        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word
        self.word_freq[word] += count

        
    def construct(self, words):
        for word in words:
            self.add_word(word)
        self.total_words = float(sum(self.word_freq.values()))
        print '{} total words with {} uniques'.format(self.total_words, len(self.word_freq))
 

    def limit_vocab_length(self, length):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            None
            
        Returns:
            None 
        """
        if length > self.__len__():
            return
        new_word_to_index = {self.unknown:0}
        new_index_to_word = {0:self.unknown}
        self.word_freq.pop(self.unknown)          #pop unk word
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        vocab_tup = sorted_tup[:length]
        self.word_freq = dict(vocab_tup)
        for word in self.word_freq:
            index = len(new_word_to_index)
            new_word_to_index[word] = index
            new_index_to_word[index] = word
        self.word_to_index = new_word_to_index
        self.index_to_word = new_index_to_word
        self.word_freq[self.unknown]=0
        
        
    def save_vocab(self, filePath):
        """
        Save vocabulary a offline file
        
        Args:
            filePath: where you want to save your vocabulary, every line in the 
            file represents a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        self.word_freq.pop(self.unknown)
        sorted_tup = sorted(self.word_freq.items(), key=operator.itemgetter(1))
        sorted_tup.reverse()
        with open(filePath, 'wb') as fd:
            for (word, freq) in sorted_tup:
                fd.write(('%s\t%d\n'%(word, freq)).encode('utf-8'))
            

    def load_vocab_from_file(self, filePath, sep='\t'):
        """
        Truncate vocabulary to keep most frequent words
        
        Args:
            filePath: vocabulary file path, every line in the file represents 
                a word with a tab seperating word and it's frequency
            
        Returns:
            None 
        """
        with open(filePath, 'rb') as fd:
            for line in fd:
                line_uni = line.decode('utf-8')
                word, freq = line_uni.split(sep)
                index = len(self.word_to_index)
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word
                self.word_freq[word] = freq
            print 'load from <'+filePath+'>, there are {} words in dictionary'.format(len(self.word_freq))
 

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    
    def decode(self, index):
        return self.index_to_word[index]

    
    def __len__(self):
        return len(self.word_to_index)

def loadId2Tag(fileName):
    """
    """
    id2tag = {}
    tag2id = {}
    with open(fileName, 'rb') as fd:
        for line in fd:
            line_uni = line.decode('utf-8')
            idTag = line_uni.split('\t')
            index = int(idTag[0])
            tag = idTag[1]
            tag = tag.strip()
            id2tag[index] = tag
            tag2id[tag]=index
    return id2tag, tag2id

def encodeNpad(dataList, vocab, trunLen=0):
    sentLen = []
    data_matrix = []
    for wordList in dataList:
        length = len(wordList)
        if trunLen !=0:
            length=min(length, trunLen)
        sentEnc = []
        if trunLen == 0:
            for word in wordList:
                sentEnc.append(vocab.encode(word))
        else:
            for i in range(trunLen):
                if i < length:
                    sentEnc.append(vocab.encode(wordList[i]))
                else:
                    sentEnc.append(vocab.encode(vocab.unknown))
        sentLen.append(length)
        data_matrix.append(sentEnc)
    return sentLen, data_matrix

def loadData(fileName, vocab, id2tag):
    '''
    Args:
        fileName: from which file to load data
        vocab: word vocabulary
        id2tag: label vocabulary
    Returns:
        ret_data: list of format(label_id, [text_ids]) already all words and label are decoded
    '''
    ret_data=[]
    with open(fileName, 'rb') as fd:
        for line in fd:
            line = line.strip()
            line_split = line.split('\t')
            label = line_split[0]
            text = line_split[1]
            label = label.decode('utf-8')
            text = text.decode('utf-8')
            label = id2tag.encode(label)
            text = [vocab.encode(o) for o in text.split()]
            ret_data.append((label, text))
    return ret_data

"""Prediction """
def pred_from_prob_single(prob_matrix):
    """

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from softmax activation
            
    Returns:
        ret: return class ids, shape of(data_num,)
    """
    ret = np.argmax(prob_matrix, axis=1)
    return ret

def pred_from_prob_multi(prob_matrix, label_num):
    """

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from softmax activation
        label_num: specify how much positive class to pick, have the shape of (data_num), type of int

    Returns:
        ret: for each case, set all positive class to 1, shape of(data_num, class_num)
    """
    order = np.argsort(prob_matrix,axis=1)
    ret = np.zeros_like(prob_matrix, np.int32)
    
    for i in range(len(label_num)):
        ret[i][order[i][-label_num[i]:]]=1
    return ret

def pred_from_prob_sigmoid(prob_matrix, threshold=0.5):
    """
    Load tag from file

    Args:
        prob_matrix: probability matrix have the shape of (data_num, class_num), 
            type of float. Generated from sigmoid activation
        threshold: when larger than threshold, consider it as true or else false
    Returns:
        ret: for each case, set all positive class to 1, shape of(data_num, class_num)
    """
    np_matrix = np.array(prob_matrix)
    ret = (np_matrix > threshold)*1
    return ret

def calculate_accuracy_single(pred_ids, label_ids):
    """
    Args:
        pred_ids: prediction id list shape of (data_num, ), type of int
        label_ids: true label id list, same shape and type as pred_ids

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if np.ndim(pred_ids) != 1 or np.ndim(label_ids) != 1:
        raise TypeError('require rank 1, 1. get {}, {}'.format(np.rank(pred_ids), np.rank(label_ids)))
    if len(pred_ids) != len(label_ids):
        raise TypeError('first argument and second argument have different length')

    accuracy = np.mean(np.equal(pred_ids, label_ids))
    return accuracy

def calculate_accuracy_multi(pred_matrix, label_matrix):
    """
    Args:
        pred_matrix: prediction matrix shape of (data_num, class_num), type of int
        label_matrix: true label matrix, same shape and type as pred_matrix

    Returns:
        accuracy: accuracy of the prediction, type float
    """
    if np.ndim(pred_matrix) != 2 or np.ndim(label_matrix) != 2:
        raise TypeError('require rank 2, 2. get {}, {}'.format(np.rank(pred_matrix), np.rank(label_matrix)))
    if len(pred_matrix) != len(label_matrix):
        raise TypeError('first argument and second argument have different length')

    match = [np.array_equal(pred_matrix[i], label_matrix[i]) for i in range(len(label_matrix))]

    return np.mean(match)

def flatten(li):
    ret = []
    for item in li:
        if isinstance(item, list) or isinstance(item, tuple):
            ret += flatten(item)
        else:
            ret.append(item)
    return ret

"""Read and make embedding matrix"""
def readEmbedding(fileName):
    """
    Read Embedding Function
    
    Args:
        fileName : file which stores the embedding
    Returns:
        embeddings_index : a dictionary contains the mapping from word to vector
    """
    embeddings_index = {}
    with open(fileName, 'r') as f:
        for line in f:
            line_uni = line.decode('utf-8')
            values = line_uni.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def mkEmbedMatrix(embed_dic, vocab_dic):
    """
    Construct embedding matrix
    
    Args:
        embed_dic : word-embedding dictionary
        vocab_dic : word-index dictionary
    Returns:
        embedding_matrix: return embedding matrix
    """
    if type(embed_dic) is not dict or type(vocab_dic) is not dict:
        raise TypeError('Inputs are not dictionary')
    if len(embed_dic) < 1 or len(vocab_dic) <1:
        raise ValueError('Input dimension less than 1')
    
    EMBEDDING_DIM = len(embed_dic.items()[0][1])
    embedding_matrix = np.zeros((len(vocab_dic) + 1, EMBEDDING_DIM), dtype=np.float32)
    for word, i in vocab_dic.items():
        embedding_vector = embed_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

"""Data iterating"""
def data_iter(data, batch_size):
    '''
    Args:
        data: list of (label_ids, [word_ids])
        batch_size: batch_size
    Returns:
        batch_x, batch_y, batch_len
    '''
    def batch_encodeNpad(data):
        '''
        Args:
            data: list of (label_ids, [word_ids])
        Returns:
            batch_x, batch_y, batch_len
        '''
        
        sent_lengths = [len(o[1]) for o in data] #b_sz
        max_sent_len = np.max(sent_lengths)
        b_sz = len(data)
        
        
        batch_x = np.zeros([b_sz, max_sent_len], dtype=np.int32)
        batch_y = np.zeros(b_sz, dtype=np.int32) #b_sz
        
        for (i, item) in enumerate(data):
            batch_y[i] = item[0]
            for (j, word_id) in enumerate(item[1]):
                batch_x[i, j] = word_id
        return  batch_x, batch_y, sent_lengths
    
    data_len = len(data)
    epoch_size = data_len // batch_size
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    for i in xrange(epoch_size):
        indices = range(i*batch_size, (i+1)*batch_size)
        indices = idx[indices]
        data_batch = [data[o] for o in indices]
        ret_batch = batch_encodeNpad(data_batch)
        yield ret_batch #(batch_x, batch_y, sent_lengths)

"""confusion calculation and logging"""
def calculate_confusion_single(pred_list, label_list, label_size):
    """Helper method that calculates confusion matrix."""
    confusion = np.zeros((label_size, label_size), dtype=np.int32)
    for i in xrange(len(label_list)):
        confusion[label_list[i], pred_list[i]] += 1
    
    tp_fp = np.sum(confusion, axis=0)
    tp_fn = np.sum(confusion, axis=1)
    tp = np.array([confusion[i, i] for i in range(len(confusion))])
    
    precision = tp.astype(np.float32)/(tp_fp+1e-40)
    recall = tp.astype(np.float32)/(tp_fn+1e-40)
    overall_prec = np.float(np.sum(tp))/(np.sum(tp_fp)+1e-40)
    overall_recall = np.float(np.sum(tp))/(np.sum(tp_fn)+1e-40)
    
    return precision, recall, overall_prec, overall_recall, confusion

def print_confusion_single(prec, recall, overall_prec, overall_recall, num_to_tag):
    """Helper method that prints confusion matrix."""
    logstr=""
    logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
    for i, tag in sorted(num_to_tag.items()):
        logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag.encode('utf-8'), prec[i], recall[i])
    logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format('OVERALL', overall_prec, overall_recall)
    logging.info(logstr)
    print logstr

def calculate_confusion_multi(pred_matrix, label_matrix):
    """Helper method that calculates confusion matrix."""
    tp = np.sum(np.logical_and(pred_matrix, label_matrix), axis=0)
    tp_fp = np.sum(pred_matrix, axis=0)
    tp_fn = np.sum(label_matrix, axis=0)
    precision = tp.astype(np.float32)/(tp_fp+1e-40)
    recall = tp.astype(np.float32)/(tp_fn+1e-40)
    overall_prec = np.float(np.sum(tp))/(np.sum(tp_fp)+1e-40)
    overall_recall = np.float(np.sum(tp))/(np.sum(tp_fn)+1e-40)
    return precision, recall, overall_prec, overall_recall

def print_confusion_multi(prec, recall, overall_prec, overall_recall, num_to_tag):
    """Helper method that prints confusion matrix."""
    logstr=""
    logstr += '{:15}\t{:7}\t{:7}\n'.format('TAG', 'Prec', 'Recall')
    for i, tag in sorted(num_to_tag.items()):
        logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format(tag.encode('utf-8'), prec[i], recall[i])
    logstr += '{:15}\t{:2.4f}\t{:2.4f}\n'.format('OVERALL', overall_prec, overall_recall)
    #logging.info(logstr)
    print logstr
