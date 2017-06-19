import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import argparse
import logging

import helper
from model import NewsModel
from Config import Config
from TfUtils import mkMask, reduce_avg

args=None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training options")
    
    parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
    parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
    parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)
    
    parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
    parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
    
    args = parser.parse_args()

    model_names={'lstm_basic': 'lstm_basic', 'cnn_basic': 'cnn_basic',
             'cbow_basic': 'cbow_basic', 'feed_back_lstm': 'feed_back_lstm'}

class Generate_Model(NewsModel):
    """Abstracts a Tensorflow graph for a learning task.

    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """
    def __init__(self, test=False, args=args):
        """options in this function"""
        self.config = Config()
        
        self.weight_Path = args.weight_path
        if args.load_config == False:
            self.config.saveConfig(self.weight_Path+'/config')
            print 'default configuration generated, please specify --load-config and run again.'
            sys.exit()
        else:
            if os.path.exists(self.weight_Path+'/config'):
                self.config.loadConfig(self.weight_Path+'/config')
            else:
                self.config.saveConfig(self.weight_Path+'/config') #if not exists config file then use default
        
        self.load_data(test)
        self.add_placeholders()
        self.add_embedding()
        inputs = self.fetch_input()
        self.logits = self.add_model(inputs)
        
        if self.config.predict_activation == 'softmax':
            self.predict_prob = tf.nn.softmax(self.logits, name='predict_probability_soft')
        elif self.config.predict_activation == 'sigmoid':
            self.predict_prob = tf.nn.sigmoid(self.logits, name='predict_probability_sig')
        """
        mean(predict_prob)

        """
        self.loss, self.train_loss = self.add_loss_op(self.logits, self.ph_input_y)
        self.train_op = self.add_train_op(self.train_loss)

    def load_data(self, test):
        self.vocab = helper.Vocab()
        self.tag_vocab = helper.Vocab()
        self.vocab.load_vocab_from_file(self.config.vocab_path, sep='\t')
        self.vocab.limit_vocab_length(self.config.vocab_size)
        self.tag_vocab.load_vocab_from_file(self.config.id2tag_path)
        self.tag_vocab.limit_vocab_length(1000)
        self.config.class_num = len(self.tag_vocab)
        if test==False:
            self.train_data = helper.loadData(self.config.train_data, self.vocab, self.tag_vocab)
            self.dev_data = helper.loadData(self.config.val_data, self.vocab, self.tag_vocab)
            step_p_epoch = len(self.train_data) // self.config.batch_size
        else:
            self.test_data = helper.loadData(self.config.test_data, self.vocab, self.tag_vocab)
            step_p_epoch=0
        self.step_p_epoch=step_p_epoch
        
    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        shape:
          self.input_placeholder   ==> (batch_size, num_steps) dtype=tf.int32
          self.label_placeholder   ==> (batch_size, class_num) dtype=tf.int32
          self.seqLen_placeholder  ==> (batch_size)            dtype=tf.int32
          self.dropout_placeholder ==> scalar                  dtype=tf.float32

        self.input_placeholder
        self.label_placeholder
        self.seqLen_placeholder
        self.dropout_placeholder
        """
        self.ph_input_x = tf.placeholder(tf.int32, (None, None), name='ph_input_x')
        self.ph_input_y = tf.placeholder(tf.int32, (None,), name='ph_input_y')
        self.ph_seqLen  = tf.placeholder(tf.int32, (None,), name='ph_seqLen')
        self.ph_drop    = tf.placeholder(tf.float32, name='ph_drop')
        self.ph_train_mode = tf.placeholder(tf.bool, name='train_mode')
        
    def create_feed_dict(self, data_batch, train_mode=True):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:

        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
              tensors created in add_placeholders.

        Args:
          data_batch: A batch of input data. (batch_x, batch_y, sent_lengths)
          train_mode: specify if you want to train or to test/validation, (to determine whether enable drop out)
        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        holder_list = [self.ph_input_x, self.ph_input_y, self.ph_seqLen, self.ph_drop, self.ph_train_mode]
        feed_list = data_batch+(self.config.dropout, train_mode)
        feed_dict = dict(zip(holder_list, feed_list))
        return feed_dict
    
    def add_embedding(self):
        if self.config.pre_trained:
            embed_dic = helper.readEmbedding(self.config.embed_path+str(self.config.embed_size))  #embedding.50 for 50 dim embedding
            embed_matrix = helper.mkEmbedMatrix(embed_dic, self.vocab.word_to_index)
            self.embedding = tf.Variable(embed_matrix, 'Embedding')
        else:
            self.embedding = tf.get_variable('Embedding', [len(self.vocab), self.config.embed_size], trainable=True)
  
    def fetch_input(self):
        inputs = tf.nn.embedding_lookup(self.embedding, self.ph_input_x) ## (batch_size, num_steps, embed_size)
        drop_input = tf.nn.dropout(inputs, keep_prob=self.config.dropout, name='drop_out')
        inputs = tf.cond(self.ph_train_mode, lambda: drop_input, lambda: inputs, name='select_input')
        return inputs
    
    def add_model(self, inputs):
        """
        Implements core of model that transforms inputs into predictions.

        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.

        Args:
          inputs: shape of (b_sz, tstp, embed_sz)
        Returns:
          out: A tensor of shape (batch_size, n_classes)
        """
        input_shape = tf.shape(inputs)
        b_sz = input_shape[0]
        tstp = input_shape[1]
        embed_sz = self.config.embed_size
        
        def basic_lstm_model(inputs):
            print "Loading basic lstm model.."
            for i in range(self.config.rnn_numLayers):
                with tf.variable_scope('rnnLayer'+str(i)):
                    lstm_cell = rnn_cell.BasicLSTMCell(self.config.hidden_size)
                    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, self.ph_seqLen,  #(b_sz, tstp, h_sz)
                                                   dtype=tf.float32 ,swap_memory=True, scope = 'badic_lstm_model_layer-'+str(i))
                    inputs = outputs #b_sz, tstp, h_sz
            mask = mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            mask = tf.expand_dims(mask, dim=2) #b_sz, tstp, 1
            
            aggregate_state = reduce_avg(outputs, mask, tf.expand_dims(self.ph_seqLen, 1), dim=-2) #b_sz, h_sz
            inputs = aggregate_state
            inputs = tf.reshape(inputs, [-1, self.config.hidden_size])
            
            for i in range(self.config.fnn_numLayers):
                inputs = rnn.rnn_cell._linear(inputs, self.config.hidden_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = rnn.rnn_cell._linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits
        
        def basic_cbow_model(inputs):
            mask = mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            mask = tf.expand_dims(mask, dim=2) #b_sz, tstp, 1
            
            aggregate_state = reduce_avg(inputs, mask, tf.expand_dims(self.ph_seqLen, 1), dim=-2) #b_sz, emb_sz
            inputs = aggregate_state
            inputs = tf.reshape(inputs, [-1, self.config.embed_size])
            
            for i in range(self.config.fnn_numLayers):
                inputs = rnn.rnn_cell._linear(inputs, self.config.embed_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = rnn.rnn_cell._linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits
                 
        def basic_cnn_model(inputs):
            in_channel = self.config.embed_size
            filter_sizes = self.config.filter_sizes
            out_channel = self.config.num_filters
            input = inputs
            for layer in range(self.config.cnn_numLayers):
                with tf.name_scope("conv-layer-"+ str(layer)):
                    conv_outputs = []
                    for i, filter_size in enumerate(filter_sizes):
                        with tf.variable_scope("conv-maxpool-%d" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, in_channel, out_channel]
                            W = tf.get_variable(name='W', shape=filter_shape)
                            b = tf.get_variable(name='b', shape=[out_channel])
                            conv = tf.nn.conv1d(                # size (b_sz, tstp, out_channel)
                              input,
                              W,
                              stride=1,
                              padding="SAME",
                              name="conv")
                            # Apply nonlinearity
                            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                            conv_outputs.append(h)
                    input = tf.concat(2, conv_outputs) #b_sz, tstp, out_channel*len(filter_sizes)
                    in_channel = out_channel * len(filter_sizes)      
            # Maxpooling 
#             mask = tf.sequence_mask(self.ph_seqLen, tstp, dtype=tf.float32) #(b_sz, tstp) 
            mask = mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            pooled = tf.reduce_max(input*tf.expand_dims(tf.cast(mask, dtype=tf.float32), 2), [1]) #(b_sz, out_channel*len(filter_sizes))
            #size (b_sz, out_channel*len(filter_sizes))
            inputs = tf.reshape(pooled, shape=[b_sz, out_channel*len(filter_sizes)])
            
            for i in range(self.config.fnn_numLayers):
                inputs = rnn.rnn_cell._linear(inputs, self.config.embed_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = rnn.rnn_cell._linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits
        
        def feed_back_lstm(inputs):
        
            def feed_back_net(inputs, seq_len, feed_back_steps):
                '''
                Args:
                    inputs: shape(b_sz, tstp, emb_sz)
                '''
                shape_of_input = tf.shape(inputs)
                b_sz = shape_of_input[0]
                h_sz = self.config.hidden_size
                tstp = shape_of_input[1]
                emb_sz = self.config.embed_size
                
                def body(time, prev_output, state_ta):
                    '''
                    Args:
                        prev_output: previous output shape(b_sz, tstp, hidden_size)
                    '''
                    
                    prev_output = tf.reshape(prev_output, shape=[-1, h_sz]) #shape(b_sz*tstp, h_sz)
                    output_linear = tf.nn.rnn_cell._linear(prev_output, output_size=h_sz, #shape(b_sz*tstp, h_sz)
                                                           bias=False, scope='output_transformer')
                    output_linear = tf.reshape(output_linear, shape=[b_sz, tstp, h_sz]) #shape(b_sz, tstp, h_sz)
                    output_linear = tf.tanh(output_linear) #shape(b_sz, tstp, h_sz)
                    
                    rnn_input = tf.concat(2, [output_linear, inputs], name='concat_output_input') #shape(b_sz, tstp, h_sz+emb_sz)
                    
                    cell = tf.nn.rnn_cell.BasicLSTMCell(h_sz)
                    cur_outputs, state = tf.nn.dynamic_rnn(cell, rnn_input, seq_len, dtype=tf.float32, swap_memory=True, time_major=False, scope = 'encoder')
                    state = tf.concat(1, state)
                    state_ta = state_ta.write(time, state)
                    return time+1, cur_outputs, state_ta #shape(b_sz, tstp, h_sz)
                
                def condition(time, *_):
                    return time < feed_back_steps
                
                state_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, clear_after_read=True, size=0)
                initial_output = tf.zeros(shape=[b_sz, tstp, h_sz], dtype=inputs.dtype, name='initial_output')
                time = tf.constant(0, dtype=tf.int32)
                _, outputs, state_ta = tf.while_loop(condition, body, [time, initial_output, state_ta], swap_memory=True)
                final_state = state_ta.read(state_ta.size()-1)
                return final_state, outputs
            
            _, outputs = feed_back_net(inputs, self.ph_seqLen, feed_back_steps=10)
            
            mask = mkMask(self.ph_seqLen, tstp) # b_sz, tstp
            mask = tf.expand_dims(mask, dim=2) #b_sz, tstp, 1
            
            aggregate_state = reduce_avg(outputs, mask, tf.expand_dims(self.ph_seqLen, 1), dim=-2) #b_sz, h_sz
            inputs = aggregate_state
            inputs = tf.reshape(inputs, [-1, self.config.hidden_size])
            
            for i in range(self.config.fnn_numLayers):
                inputs = rnn.rnn_cell._linear(inputs, self.config.hidden_size, bias=True, scope='fnn_layer-'+str(i))
                inputs = tf.nn.tanh(inputs)
            aggregate_state = inputs
            logits = rnn.rnn_cell._linear(aggregate_state, self.config.class_num, bias=True, scope='fnn_softmax')
            return logits
            
        if self.config.neural_model == model_names['lstm_basic']:
            logits = basic_lstm_model(inputs)
        elif self.config.neural_model == model_names['cbow_basic']:
            logits = basic_cbow_model(inputs)
        elif self.config.neural_model == model_names['cnn_basic']:
            logits = basic_cnn_model(inputs)
        elif self.config.neural_model == model_names['feed_back_lstm']:
            logits = feed_back_lstm(inputs)
        
        return logits

    def add_loss_op(self, logits, labels):
        """Adds ops for loss to the computational graph.

        Args:
          logits: A tensor of shape (batch_size, n_classes)
          labels: A tensor - placeholder probably,  of shape (batch_size, n_class)
        Returns:
          loss: A 0-d tensor (scalar) output
        """        
        labels = tf.one_hot(labels, self.config.class_num, on_value=1, off_value=0)
        if self.config.predict_activation == 'softmax':
            print 'Softmax activation'
            logging.info('Softmax activation')
            if self.config.single_label == False:
                scale = 1./tf.to_float(tf.reduce_sum(labels, reduction_indices=1))
                labels_float = tf.to_float(labels)* tf.expand_dims(scale, 1)
            else:
                labels_float = tf.to_float(labels)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_float)
        ########
        elif self.config.predict_activation == 'sigmoid':
            print 'Sigmoid activation'
            labels_float = tf.to_float(labels)
            logging.info('Sigmoid activation')
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits, labels_float)
        else:
            raise ValueError
        ########
        loss = tf.reduce_mean(loss)
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if v != self.embedding])
        return loss, loss + self.config.reg * reg_loss

    def add_train_op(self, loss):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.config.lr, global_step,
                                                   int(self.config.decay_epoch * self.step_p_epoch), self.config.decay_rate, staircase=True)

        #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def run_epoch(self, sess, data, verbose=10):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
            data_x: input data, have shape of (data_num, num_steps), change it to ndarray before this function is called
            data_y: label, have shape of (data_num, class_num)
            len_list: length list correspond to data_x, have shape of (data_num)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        data_len = len(data)
        total_steps =data_len // self.config.batch_size
        total_loss = []
        
        for step, data_batch in enumerate(helper.data_iter(data, self.config.batch_size)):
            feed_dict = self.create_feed_dict(data_batch, train_mode=True)
            _, loss, lr = sess.run([self.train_op, self.loss, self.learning_rate], feed_dict=feed_dict)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}, lr = {}'.format(
                    step, total_steps, np.mean(total_loss[-verbose:]), lr))
                sys.stdout.flush()
        return np.mean(total_loss)
    
    def fit(self, sess, data):
        data_len = len(data)
        total_loss = []
        for data_batch in helper.data_iter(data, self.config.batch_size):
            feed_dict = self.create_feed_dict(data_batch, train_mode=False)
            loss= sess.run(self.loss, feed_dict=feed_dict)
            total_loss.append(loss)  
        return np.mean(total_loss)
    
    def predict(self, sess, data):
        """Make predictions from the provided model.
        Args:
            sess: tf.Session()
            data_x: input data matrix have the shape of (data_num, num_steps), change it to ndarray before this function is called
            len_list: input data_length have the shape of (data_num)
        Returns:
          ret_pred_prob: Probability of the prediction with respect to each class
        """
        ret_pred_prob = []
        ret_data_batch = []
        for data_batch in helper.data_iter(data, self.config.batch_size):
            feed_dict = self.create_feed_dict(data_batch, train_mode=False)
            pred_prob = sess.run(self.predict_prob, feed_dict=feed_dict)
            ret_pred_prob.append(pred_prob)
            ret_data_batch.append(data_batch)
        ret_pred_prob = np.concatenate(ret_pred_prob, axis=0)
        return ret_pred_prob, ret_data_batch #b_sz, class_num| batch_data_x, batch_data_y, batch_lengths
#################### tibetan document classification ####################
    def ttt_predict(self, sess, data):
        """Make predictions from the provided model.
        Args:
            sess: tf.Session()
            data_x: input data matrix have the shape of (data_num, num_steps), change it to ndarray before this function is called
            len_list: input data_length have the shape of (data_num)
        Returns:
          ret_pred_prob: Probability of the prediction with respect to each class
        """
        ret_pred_prob = []
        ret_data_batch = []
        for data_batch in helper.data_iter(data, 1):
            feed_dict = self.create_feed_dict(data_batch, train_mode=False)
            pred_prob = sess.run(self.predict_prob, feed_dict=feed_dict)
            ret_pred_prob.append(pred_prob)
            ret_data_batch.append(data_batch)
        ret_pred_prob = np.concatenate(ret_pred_prob, axis=0)
        return ret_pred_prob, ret_data_batch #b_sz, class_num| batch_data_x, batch_data_y, batch_lengths

    def ttt_load_lenght(self):
        #the cumulative length of test articles
        cum_len=[]
        with open(self.config.length_path, 'rb') as fd:
            for line in fd:
                line_uni = int(line)
                cum_len.append(line_uni)
            print 'load from <'+self.config.length_path+'>, there are {} articles in test'.format(len(cum_len)-1)
        return cum_len

#################### tibetan document classification ####################

###################################################################################################

def test_case(sess, classifier, data, onset='VALIDATION'):
    print '#'*20, 'ON '+onset+' SET START ', '#'*20
    
    loss = classifier.fit(sess, data)

    if onset=='TEST':
    #     cum_len = classifier.ttt_load_lenght()
    #     new_pred=[]
    #     label_list=[]
    #
    #     pred_prob, data_batch = classifier.ttt_predict(sess, data)
    #     new_label_list = [batch_data_y for _, batch_data_y, _ in data_batch]  #b_sz, class_num
    #     new_label_list = np.concatenate(new_label_list).tolist()
    #     for i in range(len(cum_len)-1):
    #         new_pred.append(pred_prob[cum_len[i]:cum_len[i+1]].sum(axis=0))
    #         label_list.append(new_label_list[cum_len[i]])
    #     np.array(new_pred,dtype=np.float32)
    #     pred = helper.pred_from_prob_single(new_pred)

        ### For Analysis ###
        pred_prob, data_batch = classifier.ttt_predict(sess, data)
        label_list = [batch_data_y for _, batch_data_y, _ in data_batch]  #b_sz, class_num
        label_list = np.concatenate(label_list).tolist()
        pred = helper.pred_from_prob_single(pred_prob) #(data_num, )

        analysis = np.zeros([13,13])

        for i in range(len(pred)):
            analysis[label_list[i]][pred[i]] = analysis[label_list[i]][pred[i]]+1
        see = open('analysis.txt','w')
        see.writelines('\t')
        for i in range(1,13):
            see.writelines('%s\t'%(classifier.tag_vocab.index_to_word[i]))
        see.writelines('\n')
        for i in range(1,13):
            see.writelines('%s\t'%(classifier.tag_vocab.index_to_word[i]))
            for j in range(1,13):
                if (sum(analysis[i]))==0: see.writelines('%s\t'%('0'))
                else: see.writelines('%.2f\t'%(analysis[i][j]*100.0/sum(analysis[i])))
            see.writelines('\n')
        see.close()
        ### For Analysis ###

        # f1 = open(classifier.weight_Path+'/result_pre','w')
        # print(pred_prob.size)
        # print(len(pred_prob[0]))
        # f2 = open(classifier.weight_Path+'/result_lab','w')
        # f2.write(data_batch)
        # f1.close()
    else:
        pred_prob, data_batch = classifier.predict(sess, data)

        label_list = [batch_data_y for _, batch_data_y, _ in data_batch]  #b_sz, class_num
        label_list = np.concatenate(label_list).tolist()

        pred = helper.pred_from_prob_single(pred_prob) #(data_num, )

    prec, recall, overall_prec, overall_recall, _ = helper.calculate_confusion_single(
                                            pred, label_list, classifier.config.class_num)
    helper.print_confusion_single(prec, recall, overall_prec, overall_recall, classifier.tag_vocab.index_to_word)
    accuracy = helper.calculate_accuracy_single(pred, label_list)
    
    print 'Overall '+onset+' accuracy is: {}'.format(accuracy)
    print 'Overall '+onset+' loss is: {}'.format(loss)
    logging.info('Overall '+onset+' accuracy is: {}'.format(accuracy))
    logging.info('Overall '+onset+' loss is: {}'.format(loss))
    print '#'*20, 'ON '+onset+' SET END ', '#'*20
    return accuracy
    
def train_run():
    logging.info('Training start')
    with tf.Graph().as_default():
        classifier = Generate_Model()
        saver = tf.train.Saver()
        
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            best_accuracy = 0
            best_val_epoch = 0
            sess.run(tf.initialize_all_variables())
            
            for epoch in range(classifier.config.max_epochs):
                print "="*20+"Epoch ", epoch, "="*20
                loss = classifier.run_epoch(sess, classifier.train_data)
                print
                print "Mean loss in this epoch is: ", loss
                logging.info("Mean loss in {}th epoch is: {}".format(epoch, loss) )
                print '='*50
                
                if args.debug_enable:
                    test_case(sess, classifier, classifier.train_data, onset='TRAINING')
                val_accuracy = test_case(sess, classifier, classifier.dev_data, onset='VALIDATION')
                
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(classifier.weight_Path):
                        os.makedirs(classifier.weight_Path)

                    saver.save(sess, classifier.weight_Path+'/parameters.weights')
                if epoch - best_val_epoch > classifier.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
            logging.info("best val epoch is {}".format(best_val_epoch) )
            logging.info("best accuracy is {}".format(best_accuracy) )
            #####myself log best#####
    logging.info("Training complete")

def test_run():

    with tf.Graph().as_default():
        with tf.device("/gpu:" + str(args.gpu_num)):   #gpu_num options
            classifier = Generate_Model(test=True)
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, classifier.weight_Path+'/parameters.weights')
            test_case(sess, classifier, classifier.test_data, onset='TEST')

def main(_):
    logFile = args.weight_path+'/run.log'
    
    if args.train_test == "train":
        
        try:
            os.remove(logFile)
        except OSError:
            pass
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        train_run()
    else:
        logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
        test_run()

if __name__ == '__main__':
    tf.app.run()
