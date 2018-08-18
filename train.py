import os
import numpy as np 
import math
from config import Singleton
import tensorflow as tf
from Discrimiator import Dis
from Generator import Gen
from tqdm import tqdm
from dataHelper import DataHelper
FLAGS=Singleton().get_andy_flag()
helper=DataHelper(FLAGS)


g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.InteractiveSession(graph=g1)        
sess2 = tf.InteractiveSession(graph=g2)

paras=None


with g1.as_default():
    gen = Gen(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate =  0.001, 
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type="joint",
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    gen.build_pretrain()
    init1=tf.global_variables_initializer()
    saver1 = tf.train.Saver(max_to_keep=50)
    sess1.run(init1)

    
with g2.as_default():
    dis = Dis(itm_cnt = helper.i_cnt, 
             usr_cnt = helper.u_cnt, 
             dim_hidden = FLAGS.rnn_embedding_dim, 
             n_time_step = FLAGS.item_windows_size, 
             learning_rate = 0.001, #0.01
             grad_clip = 0.2,
             emb_dim = FLAGS.mf_embedding_dim,
             lamda = FLAGS.lamda,
             initdelta = 0.05,
             MF_paras=paras,
             model_type="joint",
             update_rule = 'sgd',
             use_sparse_tensor=FLAGS.sparse_tensor
             )
    dis.build_pretrain()
    init2=tf.global_variables_initializer()
    saver2 = tf.train.Saver(max_to_keep=50)
    sess2.run(init2)                

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def main(checkpoint_dir="model/"):
    
    scores = testModel(sess2,dis)
    
    log_dir = 'log/'
    helper.create_dirs(log_dir)
    dis_log = open(log_dir + 'dis_log_gan.txt', 'w')
    gen_log = open(log_dir + 'gen_log_gan.txt', 'w')

    K = 32
    for e in range(50):
        for g_epoch in range(2):    
            rewardes,pg_losses=[],[] 
                        
            for user in tqdm(helper.test_users): 
                sample_lambda,samples = 0.5,[]
                pos = helper.user_item_pos_rating_time_dict.get(user,{})   
                all_prob = softmax(gen.predictionItems(sess1,user))
                pn = (1 - sample_lambda) * all_prob
                pn[list(pos.keys())] += sample_lambda * 1.0 / len(pos)
                sample_items = np.random.choice(np.arange(helper.i_cnt), 2 * K, p=pn)
                for item in sample_items:
                    if item in list(pos.keys()):
                        pos_itm, t = item, pos[item]
                        u_seqs,i_seqs = helper.getSeqInTime(user,pos_itm,t)
                        samples.append((u_seqs,i_seqs,user,pos_itm))
                    else:
                        neg_itm, t = item, 0
                        u_seqs,i_seqs = helper.getSeqInTime(user,neg_itm,t)
                        samples.append((u_seqs,i_seqs,user,neg_itm))
                        
                u_seq_pos,i_seq_pos = [[ s[j].toarray() for s in samples ]  for j in range(2)]
                u_pos,i_pos = [[ s[j]  for s in samples ]  for j in range(2,4)]
                
                reward = dis.prediction(sess2,u_seq_pos,i_seq_pos,u_pos,i_pos,sparse=False)
                reward = (reward-np.mean(reward))/np.std(reward)
                  
                pg_loss = gen.unsupervised_train_step(sess1, u_seq_pos,i_seq_pos,u_pos,i_pos, reward)
                pg_losses.append(pg_loss)
                rewardes.append(np.sum(reward))
            print("pg loss : %.5f reward : %.5f "%(np.mean(np.array(pg_losses)),np.sum(np.array(rewardes))))                

            scores = testModel(sess1,gen)            
            buf = '\t'.join([str(x) for x in scores[1]])
            gen_log.write(str(e*2 + g_epoch) + '\t' + buf + '\n')
            gen_log.flush()
            
        for d_epoch in range(1):  
            rnn_losses,mf_losses,joint_losses=[],[],[]

            for user in tqdm(helper.test_users):                                                    
                sample_lambda,samples = 0.5,[]
                pos_dict = helper.user_item_pos_rating_time_dict.get(user,{})     
                all_prob = softmax(gen.predictionItems(sess1,user))
                pn = (1 - sample_lambda) * all_prob
                pn[list(pos_dict.keys())] += sample_lambda * 1.0 / len(pos_dict)                
                
                pos = [list(pos_dict.keys())[i] for i in np.random.choice(len(pos_dict),K)]
                neg = np.random.choice(np.arange(helper.i_cnt), size=K, p=pn)
                for i in range(len(pos)):
                    pos_itm, t = pos[i],pos_dict[pos[i]]
                    u_seqs,i_seqs = helper.getSeqInTime(user,pos_itm,t)
                    samples.append((u_seqs,i_seqs,user,pos_itm,1.))
                    
                    neg_itm, t = neg[i],0
                    u_seqs,i_seqs = helper.getSeqInTime(user,neg_itm,t)   
                    samples.append((u_seqs,i_seqs,user,neg_itm,0.))

                u_seq,i_seq = [[ s[j].toarray()  for s in samples ]  for j in range(2)]
                u,i = [[ s[j]  for s in samples ]  for j in range(2,4)]
                ratings = [ s[4]  for s in samples ] 
                
                _,loss_mf,loss_rnn,joint_loss,rnn,mf = dis.pretrain_step(sess2,ratings, u, i,u_seq,i_seq)                    
                rnn_losses.append(loss_rnn)
                mf_losses.append(loss_mf)
                joint_losses.append(joint_loss)
            print("rnn loss : %.5f mf loss : %.5f  : joint loss %.5f"%(np.mean(np.array(loss_rnn)),np.mean(np.array(loss_mf)),np.mean(np.array(joint_loss))))            
            scores = testModel(sess2,dis)
            buf = '\t'.join([str(x) for x in scores[1]])
            dis_log.write(str(e*2 + d_epoch) + '\t' + buf + '\n')
            dis_log.flush()
            
if __name__== "__main__":
    main()
