import pandas as pd 
import os
import datetime
import numpy as np 
import pickle

from tools import log_time_delta
import time
from multiprocessing import Pool
from multiprocessing import freeze_support
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
import math
from config import Singleton
import sklearn
import itertools
import tensorflow as tf
import random
from tqdm import tqdm
mp=False

class DataHelper():
    def __init__(self,conf,mode="run"):
        self.conf=conf

        self.data = self.loadData()
        
        self.train= self.data[self.data.days<0]
        self.test= self.data[self.data.days>=0]

        self.u_cnt= self.data ["uid"].max()+1 
        self.i_cnt= self.data ["itemid"].max()+1

        self.user_dict,self.item_dict=self.getdicts()

        self.users=set(self.data["uid"].unique())
        self.test_users=set(self.test["uid"].unique())
        self.items = set([i for i in range(self.i_cnt)])
        
        self.shared_users=set(self.train["uid"].unique()) & set(self.test["uid"].unique())
        
        self.image_features_dict = None
        
        get_pos_items=lambda group: set(group[group.rating>(4.99 if  self.conf.rating_flag else 0.5)]["itemid"].tolist())
        self.pos_items=self.train.groupby("uid").apply(get_pos_items)
                
        user_item_pos_rating_time_dict= lambda group:{item:time for i,(item,time)  in group[group.rating>(4.99 if  self.conf.rating_flag else 0.5)][["itemid","user_granularity"]].iterrows()}
        self.user_item_pos_rating_time_dict=self.train.groupby("uid").apply(user_item_pos_rating_time_dict).to_dict()
       
        
        self.test_pos_items=self.test.groupby("uid").apply(get_pos_items).to_dict()
        self.min_user_granularity=self.data.user_granularity.min()

            
    def create_dirs(self,dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # @log_time_delta
    def loadData(self):
        self.create_dirs("tmp")
        dataset_pkl = "tmp/"+self.conf.dataset +"_"+self.conf.split_data+("" if  self.conf.rating_flag else "_binary")+".pkl"
        if os.path.exists(dataset_pkl):
            print("data load over")
            return pickle.load(open(dataset_pkl, 'rb'))
        print("build data...")
        data_dir="data/%s"% self.conf.dataset 
        filename = os.path.join(data_dir, self.conf.train_file_name)
       
        df = pd.read_csv(filename,sep="\t", names=["uid","itemid","rating","timestamp"])

        df = df.sort_values(["uid","itemid"])

        print("there are %d users in this dataset" %(df ["uid"].max()+1))

        y,m,d = (int(i) for i in self.conf.split_data.split("-"))

        df["days"] = (pd.to_datetime(df["timestamp"]) - pd.datetime(y,m,d )).dt.days

        df["item_granularity"] = df["days"] // self.conf.item_delta   # //means floor div
        df["user_granularity"] = df["days"] // self.conf.user_delta   # //means floor div       
        

        if self.conf.threshold > 0: # remove the users while the rating of them is lower than threshold
            counts_df = pd.DataFrame(df.groupby('uid').size().rename('counts'))
            users = set(counts_df[counts_df.counts>self.conf.threshold].index)
            df = df[df.uid.isin(users)]
        if not self.conf.rating_flag :
            df["rating"]=(df["rating"]>4.99).astype('int')#movielens 3.99, neflix:4.99
            df=df[df.rating >  0.5]
#       re-arrange the user and item index from zero     
        df['u_original'] = df['uid'].astype('category')
        df['i_original'] = df['itemid'].astype('category')
        df['uid'] = df['u_original'].cat.codes
        df['itemid'] = df['i_original'].cat.codes
        df = df.drop('u_original', 1)
        df = df.drop('i_original', 1)
        
        pickle.dump(df, open(dataset_pkl, 'wb'),protocol=2)
        return df

    def user_windows_apply(self,group,user_dict):
        uid=(int(group["uid"].mode()))
        # user_dict[uid]= len(group["days"].unique())
        user_dict.setdefault(uid,{})
        for user_granularity in group["user_granularity"]:
            # print (group[group.user_granularity==user_granularity])
            if  self.conf.rating_flag:
                user_dict[uid][user_granularity]= group[group.user_granularity==user_granularity][["itemid","rating"]]
            else:
                user_dict[uid][user_granularity]= group[(group.user_granularity==user_granularity) & (group.rating>0)][["itemid","rating"]]
        return len(group["user_granularity"].unique())
    
    def item_windows_apply(self,group,item_dict):
        itemid=(int(group["itemid"].mode()))
        # user_dict[uid]= len(group["days"].unique())
        item_dict.setdefault(itemid,{})
        for item_granularity in group["item_granularity"]:
            # print (group[group.user_granularity==user_granularity])
            if  self.conf.rating_flag:
                item_dict[itemid][item_granularity]= group[group.item_granularity==item_granularity][["uid","rating"]]
            else:
                item_dict[itemid][item_granularity]= group[(group.item_granularity==item_granularity)  & (group.rating>0)][["uid","rating"]]
            # print (item_dict[itemid][item_granularity])
        return len(group["item_granularity"].unique())
            
    # @log_time_delta
    def getdicts(self):
 
        dict_pkl = "tmp/user_item_"+self.conf.dataset+("" if  self.conf.rating_flag else "_binary")+".pkl"
        if os.path.exists(dict_pkl):
            start=time.time()
            import gc
            gc.disable()
            user_dict,item_dict= pickle.load(open(dict_pkl, 'rb'))
            gc.enable()
            print( "load dict cost  time: %.5f "%( time.time() - start))
        else:            
            print("build data...")
            user_dict,item_dict={},{}
            user_windows = self.data.groupby("uid").apply(self.user_windows_apply,user_dict=user_dict)
            item_windows = self.data.groupby("itemid").apply(self.item_windows_apply,item_dict=item_dict)
            pickle.dump([user_dict,item_dict], open(dict_pkl, 'wb'),protocol=2)

        return user_dict,item_dict
    



    def getSeqInTime(self,userid,itemid,chosen_t=0, choice_type="nothing"):
        if choice_type=="given":
            pos_items_time_dict=self.user_item_pos_rating_time_dict.get(userid,{})
            chosen_t=pos_items_time_dict.get(itemid)
        if choice_type=="random":
            chosen_t= random.choice (range(self.min_user_granularity +self.conf.user_windows_size,0))
        if choice_type=="best" :
            u_seqss,i_seqss= self.getSeqOverAlltime(user,neg_item_id)
            predicted = model.prediction(sess,u_seqss,i_seqss, [user]*len(u_seqss),[neg_item_id]*len(u_seqss),sparse=True)
            index=np.argmax(predicted)
            return (u_seqss[index],i_seqss[index])

        u_seqs,i_seqs=[],[]
        for i in range(chosen_t-self.conf.user_windows_size,chosen_t):
            u_seqs.append(self.user_dict[userid].get(i,None))
            i_seqs.append(self.item_dict[itemid].get(i,None))
        if self.conf.is_sparse:
            return self.getUserVector(u_seqs),self.getItemVector(i_seqs)
        else:

            return self.getUserVector_raw(u_seqs),self.getItemVector_raw(i_seqs)

    def getSeqOverAlltime(self,userid, itemid):  

        u_seqs,i_seqs=[],[]
        for t in range(self.data["user_granularity"].min(),0):
    
            u_seqs.append(self.user_dict[userid].get(t,None))
            i_seqs.append(self.item_dict[itemid].get(t,None))

        u_seqss,i_seqss=[],[]
        for t in range( self.data["user_granularity"].min() ,0- self.conf.user_windows_size):
            u_seqss.append( u_seqs[t:t+self.conf.user_windows_size])
            i_seqss.append( i_seqs[t:t+self.conf.user_windows_size])     

        if self.conf.is_sparse:
            return [i for i in map(self.getUserVector, u_seqss)],[i for i in map(self.getItemVector, i_seqss)]
        else:              
            return [i for i in map(self.getUserVector_raw, u_seqss)],[i for i in map(self.getItemVector_raw, i_seqss)]
        

    def prepare_balance_pair(self,pool=None,sess=None,model=None, mode="train", epoches_size=1,shuffle=True,fresh=False,users=None):
        if users is None:
            users=self.train.uid.unique()
        samples=[]
        for user in  tqdm(users):
            pos_items= self.pos_items.get(user,[])
            candidates = list( set(range(self.i_cnt)) - set(pos_items) )   
            pos_items=list(pos_items)
            if self.conf.dns:
                all_rating = model.predictionItems(sess,user)                           # todo delete the pos ones            
                exp_rating = np.exp(np.array(all_rating) *self.conf.temperature)
                prob = exp_rating / np.sum(exp_rating)            
                # negative_items_argmax = np.argsort(prob)[::-1][:2]
                neg_items=np.random.choice(np.arange(self.i_cnt), size=len(self.pos_items_time_dict), p=prob)
            else:        
                neg_items= np.random.choice(candidates,len(pos_items))
            for i in range(len(pos_items)):
                u_seqs,pos_item_seq=self.getSeqInTime(user,pos_items[i],choice_type="given" )
                u_seqs,neg_item_seq=self.getSeqInTime(user,neg_items[i], choice_type="random")  #best
                if self.conf.pairwise:
                    sample =   (user,u_seqs,pos_items[i],pos_item_seq,neg_items[i],neg_item_seq)
                    samples.append(sample)
                else:
                    samples.append((user,u_seqs,pos_items[i],pos_item_seq,1))
                    samples.append((user,u_seqs,neg_items[i],neg_item_seq,0))
        return samples

    
    def getBatch_with_multi_pickle(self,pool=None,dns=True,sess=None,model=None,fresh=True,mode="train", epoches_size=1,shuffle=True,pickle_name=None,samples=None):
        users=self.train.uid.unique()
        pickle_path = "tmp/samples_"+ ("dns" +str(self.conf.subset_size)+"_" if dns else "uniform") + ("_pair" if self.conf.pairwise else "") +("_sparse_tensor_" if self.conf.sparse_tensor else ( "_sparse" if self.conf.is_sparse else "_") ) +self.conf.dataset+"_"+str(self.conf.user_windows_size)+("" if  self.conf.rating_flag else "_binary")  +mode +("" if  self.conf.user_windows_size==4 else "_seq"+str(self.conf.user_windows_size))
        if not os.path.exists(pickle_path):
            print("No pickled samples here, need to be created")
            self.create_dirs(pickle_path)
            groups = [users[i:i+1000] for i in range(0,len(users),1000)]
            for i,group in enumerate(groups):
                samples=self.prepare_balance_pair(users=group,mode=mode, sess=sess,model=model, epoches_size=epoches_size)
                pickle_name=os.path.join(pickle_path,str(i))
                pickle.dump(samples, open(pickle_name, 'wb'),protocol=2)
        
        for i in os.listdir(pickle_path):
            if os.path.isfile(os.path.join(pickle_path,i)):
                
                pickle_name=os.path.join(pickle_path,i)
                print("load samples from file %s" % pickle_name)
                import gc
                gc.disable()
                samples=pickle.load(open(pickle_name, 'rb'))
                gc.enable()
                samples=[sample for sample in samples if sample[0] in self.test.uid.unique()]
                print("process %d samples" % len(samples))
                for batch in self.getBatch(samples=samples,pool=pool,dns=dns,sess=sess,model=model,mode=mode, epoches_size=epoches_size,shuffle=shuffle,pickle_name=pickle_name):
                    yield batch
    
    def getBatch(self,pool=None,dns=True,sess=None,model=None,fresh=True,mode="train", epoches_size=1,shuffle=True,pickle_name=None,samples=None):
        
        if samples is None:
            if pickle_name==None:
                pickle_name = "tmp/samples_"+ ("dns" +str(self.conf.subset_size)+"_" if dns else "uniform") + ("_pair" if self.conf.pairwise else "") +("_sparse_tensor_" if self.conf.sparse_tensor else ( "_sparse" if self.conf.is_sparse else "_") ) +self.conf.dataset+"_"+str(self.conf.user_windows_size)+("" if  self.conf.rating_flag else "_binary")  +mode+".pkl"
                
            if os.path.exists(pickle_name) and not fresh:
                import gc
                gc.disable()
                print (pickle_name)
                samples=pickle.load(open(pickle_name, 'rb'))
                gc.enable()        
            else:            
                samples = self.prepare_balance_pair(mode=mode, sess=sess,model=model, epoches_size=epoches_size)            
                pickle.dump(samples, open(pickle_name, 'wb'),protocol=2)
                
       
        start=time.time()
        random.shuffle(samples)                      
        print("shuffle time spent %f"% (time.time()-start))
        
        n_batches = int(len(samples)/ self.conf.batch_size)
        print("%d batch"% n_batches)
                
        for i in range(0,n_batches):
            start=time.time()            
           
            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
                
            if not self.conf.pairwise:
                u_seqs=[pair[1] for pair in batch]
                i_seqs=[pair[3] for pair in batch]
                if not self.conf.sparse_tensor and self.conf.is_sparse:
                    if pool is not None:
                        u_seqs=pool.map(sparse2dense, u_seqs)
                        i_seqs=pool.map(sparse2dense, i_seqs)
                    else:
                        u_seqs=[v for v in map(sparse2dense, u_seqs)]
                        i_seqs=[v for v in map(sparse2dense, i_seqs)]


                ratings=[pair[4] for pair in batch]
                userids=[pair[0] for pair in batch]
                itemids=[pair[2] for pair in batch]

                if self.conf.sparse_tensor:
                    u_seqs,i_seqs=self.get_sparse_intput(u_seqs,i_seqs)
                yield u_seqs,i_seqs,ratings,userids,itemids
            else:
                
                user=[pair[0] for pair in batch]
                u_seqs=[pair[1] for pair in batch]
                item=[pair[2] for pair in batch]
                i_seqs=[pair[3] for pair in batch]
                item_neg=[pair[4] for pair in batch]
                i_seqs_neg=[pair[5] for pair in batch]
                if not self.conf.sparse_tensor and self.conf.is_sparse:
                    if pool is not None:
                        u_seqs=pool.map(sparse2dense, u_seqs)
                        i_seqs=pool.map(sparse2dense, i_seqs)
                        item_neg=pool.map(sparse2dense, item_neg)
                    else:
                        u_seqs=[v for v in map(sparse2dense, u_seqs)]
                        i_seqs=[v for v in map(sparse2dense, i_seqs)]
                        i_seqs_neg=[v for v in map(sparse2dense, i_seqs_neg)]
                if self.conf.sparse_tensor:
                    u_seqs=self.get_user_sparse_input(u_seqs)
                    i_seqs=self.get_item_sparse_input(i_seqs)
                    i_seqs_neg=self.get_item_sparse_input(i_seqs_neg)

                yield (user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)
 
    
    def getBatch_with_Files(self,pool=None,dns=True,sess=None,model=None,fresh=True,mode="train", epoches_size=1,shuffle=True):
        
        pickle_name = "tmp/samples_"+ ("dns" +str(self.conf.subset_size)+"_" if dns else "uniform") + ("_pair" if self.conf.pairwise else "") +("_sparse_tensor_" if self.conf.sparse_tensor else ( "_sparse" if self.conf.is_sparse else "_") ) +self.conf.dataset+"_"+str(self.conf.user_windows_size)+("" if  self.conf.rating_flag else "_binary")  +mode+".pkl"
        print (pickle_name)
        if os.path.exists(pickle_name) and not fresh:
            import gc
            gc.disable()
            samples=pickle.load(open(pickle_name, 'rb'))
            gc.enable()        
        else:
            samples = self.prepare_balance_pair(mode=mode, sess=sess,model=model, epoches_size=epoches_size)            
            pickle.dump(samples, open(pickle_name, 'wb'),protocol=2)

        start=time.time()
        random.shuffle(samples)                      
        print("shuffle time spent %f"% (time.time()-start))

        n_batches = int(len(samples)/ self.conf.batch_size)
        print("%d batch"% n_batches)
                
        for i in range(0,n_batches):
            start=time.time()            
           
            batch = samples[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
                
            if not self.conf.pairwise:
                u_seqs=[pair[1] for pair in batch]
                i_seqs=[pair[3] for pair in batch]
                if not self.conf.sparse_tensor and self.conf.is_sparse:
                    if pool is not None:
                        u_seqs=pool.map(sparse2dense, u_seqs)
                        i_seqs=pool.map(sparse2dense, i_seqs)
                    else:
                        u_seqs=[v for v in map(sparse2dense, u_seqs)]
                        i_seqs=[v for v in map(sparse2dense, i_seqs)]


                ratings=[pair[4] for pair in batch]
                userids=[pair[0] for pair in batch]
                itemids=[pair[2] for pair in batch]

                if self.conf.sparse_tensor:
                    u_seqs,i_seqs=self.get_sparse_intput(u_seqs,i_seqs)
                yield u_seqs,i_seqs,ratings,userids,itemids
            else:
                #(user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)
                user=[pair[0] for pair in batch]
                u_seqs=[pair[1] for pair in batch]
                item=[pair[2] for pair in batch]
                i_seqs=[pair[3] for pair in batch]
                item_neg=[pair[4] for pair in batch]
                i_seqs_neg=[pair[5] for pair in batch]
                if not self.conf.sparse_tensor and self.conf.is_sparse:
                    if pool is not None:
                        u_seqs=pool.map(sparse2dense, u_seqs)
                        i_seqs=pool.map(sparse2dense, i_seqs)
                        item_neg=pool.map(sparse2dense, item_neg)
                    else:
                        u_seqs=[v for v in map(sparse2dense, u_seqs)]
                        i_seqs=[v for v in map(sparse2dense, i_seqs)]
                        i_seqs_neg=[v for v in map(sparse2dense, i_seqs_neg)]
                if self.conf.sparse_tensor:
                    u_seqs=self.get_user_sparse_input(u_seqs)
                    i_seqs=self.get_item_sparse_input(i_seqs)
                    i_seqs_neg=self.get_item_sparse_input(i_seqs_neg)

                yield (user,u_seqs,item,i_seqs,item_neg,i_seqs_neg)
 
      

    def getUserVector_raw(self,user_sets):
        u_seqs=[]
        for user_set in user_sets:
            u_seq=[0]*(self.i_cnt)
       
            if not user_set is None:
                for index,row in user_set.iterrows():
                    u_seq[row["itemid"]]=row["rating"]
            u_seqs.append(u_seq)
        return np.array(u_seqs)
    
    
    def getItemVector_raw(self,item_sets):
        i_seqs=[]
        for item_set in item_sets:
            i_seq=[0]*(self.u_cnt)
            if not item_set is None:
                for index,row in item_set.iterrows():
                   i_seq[row["uid"]]=row["rating"]
            i_seqs.append(i_seq)
        return np.array(i_seqs)
    def getItemVector(self,item_sets):
        rows=[]
        cols=[]
        datas=[]
        for index_i,item_set in enumerate(item_sets):
            if not item_set is None:
                for index_j,row in item_set.iterrows():
                    rows.append(index_i)
                    cols.append(row["uid"])
                    datas.append(row["rating"])
        if self.conf.sparse_tensor:
            return ( rows,cols ,datas)
        result=csr_matrix((datas, (rows, cols)), shape=(self.conf.user_windows_size, self.u_cnt))
        return result
    
    def getUserVector(self,user_sets):
        rows=[]
        cols=[]
        datas=[]
        for index_i,user_set in enumerate(user_sets):           
            if not user_set is None:
                for index,row in user_set.iterrows():
                    rows.append(index_i)
                    cols.append(row["itemid"])
                    datas.append(row["rating"])
        if self.conf.sparse_tensor:
            return ( rows,cols ,datas)
        return csr_matrix((datas, (rows, cols)), shape=(self.conf.user_windows_size, self.i_cnt))

    def getBatch4MF(self,flag="train",shuffle=True):
        np.random.seed(0)
        train_flag= np.random.random(len(self.data))>0.2
        if flag=="train":
            df=self.data[train_flag]
            if shuffle ==True:
                df=df.iloc[np.random.permutation(len(df))]
                print ("shuffle over")
        else:
            df=self.data[~train_flag]

        n_batches= int(len(df)/ self.conf.batch_size)
        for i in range(0,n_batches):
            batch = df[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
            yield batch["uid"],batch["itemid"],batch["rating"]
        batch= df[-1*self.conf.batch_size:] 
        yield batch["uid"],batch["itemid"],batch["rating"]

    def testModel(self,sess,discriminator,flag="test"):
        results=np.array([])
        for uid,itemid,rating in self.getBatch4MF(flag=flag):
            feed_dict={discriminator.u: uid, discriminator.i: itemid}
            predicted = sess.run(discriminator.pre_logits,feed_dict=feed_dict)
            error=(np.array(predicted)-np.array(rating))
            se= np.square(error)
            results=np.append(results,se)
        mse=np.mean(results)
        return math.sqrt(mse)

    def evaluateRMSE(self,sess,model):
        results=np.array([])
        for u_seqss,i_seqss,ratingss,useridss,itemidss in self.getDataWithSeq(mode="test",rating_flag=True):
            predicted = model.prediction(sess, u_seqss, i_seqss, useridss, itemidss)
            # print(predicted)
            # print(ratingss)
            error=(np.array(predicted)*5-np.array(ratingss))  # different optimitic indicator
            se= np.square(error)
            results=np.append(results,se)
        mse=np.mean(results)

        return math.sqrt(mse)
    
    def getDataWithSeq(self,shuffle=True,mode="train",epoches=2,rating_flag=False):

        if True:
        # try:     
            if mp:    
                pool= Pool(cpu_count())
            else:
                pool=None
            samples=self.prepare_uniform(pool,mode=mode, epoches_size=1)
            batches=samples
            for i in range(epoches):
                if mode=="train" and shuffle:      
                    batches =sklearn.utils.shuffle(batches)

                n_batches= int(len(batches)/ self.conf.batch_size)
                for i in range(0,n_batches):
                    batch = batches[i*self.conf.batch_size:(i+1) * self.conf.batch_size]
                    if mp:
                        u_seqs=pool.map(sparse2dense, [ii[0] for ii in batch])
                        i_seqs=pool.map(sparse2dense, [ii[1] for ii in batch])
                    else:
                        u_seqs=[record for record in map(sparse2dense, [ii[0] for ii in batch])]
                        i_seqs=[record for record in map(sparse2dense, [ii[1] for ii in batch])]
                    if rating_flag:
                        ratings=[int(ii[2]) for ii in batch]
                    else:
                        ratings=[int(ii[2]>(4.99 if  self.conf.rating_flag else 0.5)) for ii in batch]
                    userids=[ii[3] for ii in batch]
                    itemids=[ii[4] for ii in batch]
                    yield u_seqs,i_seqs,ratings,userids,itemids
        if mp:
            pool.close()


        
    def getTestFeedingData(self,userid, rerank_indexs):
        u_seqs=[]
        for t in range(-1*self.conf.user_windows_size,0):
            u_seqs.append(self.user_dict[userid].get(t,None))
        i_seqss=[]
        for itemid in rerank_indexs:
            i_seqs=[]
            for t in range(-1*self.conf.user_windows_size,0):
                i_seqs.append(self.item_dict[itemid].get(t,None))
            i_seqss.append(i_seqs)
        return self.getUserVector(u_seqs),[i for i in map(self.getItemVector, i_seqss)]
  
  
    def evaluateMultiProcess(self,sess,model,mp=False,users_set=None):
        if users_set is None:
            users_set=self.test_users
        print("evaluate %d users" %len(users_set))
        results=None
        if mp:
            pool=Pool(cpu_count())
            results= pool.map(self.getScore,zip(list(users_set), itertools.repeat(sess),itertools.repeat(model) ))
        else:
            results= [ i for i in map(self.getScore,zip(users_set, itertools.repeat(sess),itertools.repeat(model) ))]


        return list(np.mean(np.array(results),0))
    def get_user_sparse_input(self,user_sequence):
        _indices,_values=[],[]
        for index,(cols,rows,values)  in enumerate(user_sequence):
            _indices.extend([index,x,y]  for x,y in zip(cols,rows) )   #sorted(zip(cols,rows),key =lambda x:x[0]*2000+x[1] )
            _values.extend(values)    
        if len(_indices)==0:
            return ([[0,0,0]],[0],[len(user_sequence),self.conf.user_windows_size,self.i_cnt ])        
        user_input= (_indices,_values,[len(user_sequence),self.conf.user_windows_size,self.i_cnt ])
        return user_input
    def get_item_sparse_input(self,item_sequence):
        _indices,_values=[],[]
        for index,(cols,rows,values)  in enumerate(item_sequence):
            _indices.extend([index,x,y]  for x,y in  zip(cols,rows))
            _values.extend(values)
        if len(_indices)==0:
            return ([[0,0,0]],[0],[len(item_sequence),self.conf.user_windows_size,self.u_cnt ])
        item_input= (_indices,_values,[len(item_sequence),self.conf.user_windows_size,self.u_cnt ])
        return item_input
    def get_sparse_intput(self,user_sequence,item_sequence):
        user_input=self.get_user_sparse_input(user_sequence)
        item_input=self.get_item_sparse_input(item_sequence)
        
        return user_input,item_input




    def getScore(self,args):
        
        rerank=True
        (user_id,sess,model)=args
        
        if model is None:
            print ("there is no model, it is random guessing instead!")
            all_rating= np.random.random( len(self.items)+1)  #[user_id]        
        else:
            all_rating = model.predictionItems(sess,user_id)[0] # MF rating

        candiate_index = self.items - self.pos_items.get(user_id, set())
        scores =[ (index,all_rating[index]) for index in candiate_index ]

        sortedScores = sorted(scores ,key= lambda x:x[1], reverse = True )

        pre_rank_list= [1 if ii[0] in self.test_pos_items.get(user_id, set()) else 0 for ii in sortedScores[:10]]
        pre_result = getResult(pre_rank_list)

        if not rerank or self.conf.model_type=="mf":
            return pre_result


        rerank_indexs= ([ii[0] for ii in sortedScores[:self.conf.re_rank_list_length]])
        u_seqs,i_seqss=self.getTestFeedingData(user_id, rerank_indexs)

        
        if model is None:
            print ("there is no model, it is random guessing instead!")
            scores=np.random.random( len(rerank_indexs))
        else:

            if self.conf.use_cnn:                
                img_feats = [self.image_features_dict.get(i,[0]*2048)for i in rerank_indexs]            
                scores = model.prediction(sess,[u_seqs] * self.conf.re_rank_list_length, i_seqss , 
                                          [user_id] * self.conf.re_rank_list_length, rerank_indexs,True,False,img_feats)
            else:
                scores = model.prediction(sess,[u_seqs] * self.conf.re_rank_list_length, i_seqss , [user_id] * self.conf.re_rank_list_length, rerank_indexs,use_sparse_tensor=False)

        
        sortedScores = sorted(zip(rerank_indexs,scores) ,key= lambda x:x[1], reverse = True )
        rank_list= [1 if ii[0] in self.test_pos_items.get(user_id, set()) else 0 for ii in sortedScores[:10]]

        result = getResult(rank_list)
        # print(rank_list)
        # print("rerank score: %s"%(str(result-pre_result)))
        return pre_result,result


def sparse2dense(sparse):
    return sparse.toarray()

def getResult(r):

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)
    mrr =reciprocal_rank(r)
    ap = average_precision(r)
    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10, mrr,ap])
def reciprocal_rank(r):
    nonzero_list=np.asarray(r).nonzero()[0]
    return 1. / (nonzero_list[0] + 1) if nonzero_list.size else 0

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k([1]* k,k)
#    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)
def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)
