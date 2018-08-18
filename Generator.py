from __future__ import division

import tensorflow as tf
import pickle

class Gen(object):
    def __init__(self, itm_cnt, usr_cnt, dim_hidden, n_time_step, learning_rate, grad_clip, emb_dim, lamda=0.2, initdelta=0.05,MF_paras=None,model_type="rnn",use_sparse_tensor=True,update_rule='adam'):

        self.V_M = itm_cnt
        self.V_U = usr_cnt
        self.param=MF_paras
        self.H = dim_hidden
        self.T = n_time_step
        
        self.MF_paras=MF_paras
        self.grad_clip = grad_clip

        self.weight_initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)

        
        self.sparse_tensor=use_sparse_tensor
        
        if self.sparse_tensor:
            self.user_sparse_tensor= tf.sparse_placeholder(tf.float32)
            self.user_sequence = tf.sparse_tensor_to_dense(self.user_sparse_tensor)
            self.item_sparse_tensor= tf.sparse_placeholder(tf.float32)
            self.item_sequence = tf.sparse_tensor_to_dense(self.item_sparse_tensor)   
            
        else:

            self.item_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_U])        
            self.user_sequence = tf.placeholder(tf.float32, [None, self.T, self.V_M])   

        self.rating = tf.placeholder(tf.float32, [None,])
                        
        self.learning_rate = learning_rate
    
    
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.initdelta = initdelta
        
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)

        self.paras_rnn=[]
        self.model_type=model_type
        self.update_rule = update_rule
        
    def _init_MF(self):
        with tf.variable_scope('MF'):
            if self.MF_paras is None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.V_U, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.V_M, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.V_M]))            
                self.user_bias = tf.Variable(tf.zeros([self.V_U])) 
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.user_bias = tf.Variable(self.param[2])           
                self.item_bias = tf.Variable(self.param[3])     

            self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
            self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
            self.i_bias = tf.gather(self.item_bias, self.i)
            self.u_bias = tf.gather(self.user_bias, self.u)

            self.paras_mf=[self.user_embeddings,self.item_embeddings,self.user_bias,self.item_bias]

    def _decode_lstm(self, h_usr, h_itm, reuse=False):
        if False:
            with tf.variable_scope('G_rating', reuse=reuse):
                w_usr = tf.get_variable('w_usr', [self.H, self.H], initializer=self.weight_initializer)
                b_usr = tf.get_variable('b_usr', [self.H], initializer=self.const_initializer)
                w_itm = tf.get_variable('w_itm', [self.H, self.H], initializer=self.weight_initializer)
                b_itm = tf.get_variable('b_itm', [self.H], initializer=self.const_initializer)

                usr_vec = tf.matmul(h_usr, w_usr) + b_usr
                
                itm_vec = tf.matmul(h_itm, w_itm) + b_itm
                                        
                out_preds = tf.reduce_sum(tf.multiply(usr_vec, itm_vec), 1)                      
                self.paras_rnn.extend([w_usr,b_usr,w_itm,b_itm])
                return out_preds
        else:
            out_preds = tf.reduce_sum(tf.multiply(h_usr, h_itm), 1) 
            print("Do not use a fully-connectted layer at the time of output decoding.")  
            return out_preds
            
    def _get_initial_lstm(self, batch_size):
        with tf.variable_scope('G_initial_lstm'):                        
            c_itm = tf.zeros([batch_size, self.H], tf.float32)
            h_itm = tf.zeros([batch_size, self.H], tf.float32)
            c_usr = tf.zeros([batch_size, self.H], tf.float32)
            h_usr = tf.zeros([batch_size, self.H], tf.float32) 
            # self.paras_rnn.extend([c_itm, h_itm, c_usr, h_usr])   # these variable should be trainable or not                     
            return c_itm, h_itm, c_usr, h_usr

    def _item_embedding(self, inputs, reuse=False):
        with tf.variable_scope('G_item_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V_U, self.H], initializer=self.emb_initializer)           
            x_flat = tf.reshape(inputs, [-1, self.V_U]) #(N * T, U)       
            x = tf.matmul(x_flat, w) #(N * T, H)
            x = tf.reshape(x, [-1, self.T, self.H]) #(N, T, H)
            self.paras_rnn.extend([w])
            return x
        
    def _user_embedding(self, inputs, reuse=False):
        with tf.variable_scope('G_user_embedding', reuse=reuse):
           w = tf.get_variable('w', [self.V_M, self.H], initializer=self.emb_initializer)           
           x_flat = tf.reshape(inputs, [-1, self.V_M]) #(N * T, M)       
           x = tf.matmul(x_flat, w) #(N * T, H)
           x = tf.reshape(x, [-1, self.T, self.H]) #(N, T, H)
           self.paras_rnn.extend([w])
           return x
        
    def build_pretrain(self):
        
        batch_size = tf.shape(self.item_sequence)[0]
                       
        c_itm, h_itm, c_usr, h_usr = self._get_initial_lstm(batch_size)
        x_itm = self._item_embedding(inputs=self.item_sequence)
        x_usr = self._user_embedding(inputs=self.user_sequence)    

        itm_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)
        usr_lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)
        
        self._init_MF()
        
        for t in range(self.T):
            with tf.variable_scope('G_itm_lstm', reuse=(t!=0)):
                _, (c_itm, h_itm) = itm_lstm_cell(inputs=x_itm[:,t,:], state=[c_itm, h_itm])            
            with tf.variable_scope('G_usr-lstm', reuse=(t!=0)):
                _, (c_usr, h_usr) = usr_lstm_cell(inputs=x_usr[:,t,:], state=[c_usr, h_usr])
              
        self.item_bias_rnn = tf.Variable(tf.zeros([self.V_M]))
        self.i_bias_rnn = tf.gather(self.item_bias_rnn, self.i)
        self.logits_RNN = self._decode_lstm(h_usr, h_itm, reuse=False) +self.i_bias_rnn    
        
        self.loss_RNN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.logits_RNN)) #+ RNN_Regularizer
        self.logits_MF = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias  +self.u_bias       
        self.loss_MF = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.logits_MF))
        
        self.pre_joint_logits = self.logits_MF + self.logits_RNN
        
        self.joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rating, logits=self.pre_joint_logits))
        
        self.joint_loss += self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.u_bias) +tf.nn.l2_loss(self.i_bias))
        self.joint_loss += self.lamda * tf.reduce_sum([tf.nn.l2_loss(para) for para in self.paras_rnn])

        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer
        else:
            self.optimizer = tf.train.GradientDescentOptimizer
            
        optimizer = self.optimizer(learning_rate=self.learning_rate)
        if self.model_type == 'joint':
            grads = tf.gradients(self.joint_loss, tf.trainable_variables())
        elif self.model_type == 'rnn':
            grads = tf.gradients(self.loss_RNN, tf.trainable_variables())
        elif self.model_type == 'mf':
            grads = tf.gradients(self.loss_MF, tf.trainable_variables())
                    
            
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        clipped_gradients = [(tf.clip_by_value(_[0], -self.grad_clip, self.grad_clip), _[1]) for _ in grads_and_vars if _[1] is not None and _[0] is not None]
        self.pretrain_updates = optimizer.apply_gradients(grads_and_vars=clipped_gradients)            
            
        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias +self.u_bias

        self.reward = tf.placeholder(tf.float32)
        if self.model_type == 'rnn':
            self.pg_loss = - tf.reduce_mean(tf.log( tf.sigmoid(self.logits_RNN)) * self.reward)          
            self.pg_loss += self.lamda * tf.reduce_sum([tf.nn.l2_loss(para) for para in self.paras_rnn])
        else:
            self.pg_loss = - tf.reduce_mean(tf.log( tf.sigmoid(self.pre_joint_logits)) * self.reward)          
            self.pg_loss += self.lamda * (tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.u_bias) +tf.nn.l2_loss(self.i_bias))
            self.pg_loss += self.lamda * tf.reduce_sum([tf.nn.l2_loss(para) for para in self.paras_rnn])
            
        pg_grads = tf.gradients(self.pg_loss, tf.trainable_variables())               
        pg_grads_and_vars = list(zip(pg_grads, tf.trainable_variables()))
        self.pg_updates = optimizer.apply_gradients(grads_and_vars=pg_grads_and_vars)                                            
        
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias
        self.all_prob = tf.nn.softmax(self.all_rating)
        
    def pretrain_step(self, sess,  rating, u, i,user_sequence=None, item_sequence=None): 
        if user_sequence is not None:
            if self.sparse_tensor:
                 outputs = sess.run([self.pretrain_updates, self.loss_MF ,self.loss_RNN,self.joint_loss,self.logits_MF,self.logits_RNN ], feed_dict = {self.user_sparse_tensor: user_sequence, 
                                self.item_sparse_tensor: item_sequence, self.rating: rating, self.u: u, self.i: i})
            else:
                outputs = sess.run([self.pretrain_updates, self.loss_MF ,self.loss_RNN,self.joint_loss,self.logits_MF,self.logits_RNN ], feed_dict = {self.user_sequence: user_sequence, 
                                self.item_sequence: item_sequence, self.rating: rating, self.u: u, self.i: i})
        else:
            outputs = sess.run([self.pretrain_updates, self.joint_loss,self.pre_logits_MF], feed_dict = {self.rating: rating, self.u: u, self.i: i})

        return outputs

    def prediction(self, sess, user_sequence, item_sequence, u, i,sparse=True, use_sparse_tensor = None):
        if self.sparse_tensor and (use_sparse_tensor is None or use_sparse_tensor!=False):          

            outputs = sess.run(self.pre_joint_logits, feed_dict = {self.user_sparse_tensor: user_sequence, 
                        self.item_sparse_tensor: item_sequence, self.u: u, self.i: i})  
            return outputs
        if sparse:
            user_sequence,item_sequence=[ii.toarray() for ii in user_sequence],[ii.toarray() for ii in item_sequence]
        outputs = sess.run(self.pre_joint_logits, feed_dict = {self.user_sequence: user_sequence, 
                        self.item_sequence: item_sequence, self.u: u, self.i: i})  
        return outputs
    
    def predictionItems(self, sess, u):
        outputs = sess.run(self.all_logits, feed_dict = {self.u: u})  
        return outputs
    
    def predictionBatchUser(self, sess, u):
        outputs = sess.run(self.all_prob, feed_dict = {self.u: u})  
        return outputs
    
    def unsupervised_train_step(self,sess, user_sequence, item_sequence, u, i,reward):

        _,pg_loss = sess.run([self.pg_updates , self.pg_loss], feed_dict = {self.user_sequence: user_sequence, 
                            self.item_sequence: item_sequence,  self.u: u, self.i: i ,self.reward:reward})
        return pg_loss
   

    def saveMFModel(self, sess, filename):
        self.paras_mf = [self.user_embeddings,self.item_embeddings,self.user_bias,self.item_bias]
        param = sess.run(self.paras_mf)
        pickle.dump(param, open(filename, 'wb'))


    