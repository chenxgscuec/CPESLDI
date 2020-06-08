# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:20:24 2020

@author: Administrator
"""
import os,sys
from feamodule import CTD
from feamodule import ProtParam as PP
from feamodule import ORF_length as ORF_len
import Bio.SeqIO as Seq
from feamodule import fickett
from feamodule  import FrameKmer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score,confusion_matrix,roc_curve,auc,matthews_corrcoef,precision_recall_fscore_support
import random
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
#from imblearn.over_sampling import SMOTE,ADASYN,RandomOverSampler


#You can replace it with your own path here
train_lncRNA = "/mnt/ls1/data/CPPred_data/Human/Homo38.ncrna_training.fa"
train_pcts = "/mnt/ls1/data/CPPred_data/Human/Human.coding_RNA_training.fa"

test_1_lncRNA = "/mnt/ls1/data/CPPred_data/Human/Homo38_ncrna_test.fa"
test_1_pcts = "/mnt/ls1/data/CPPred_data/Human/Human_coding_RNA_test.fa"
test_2_lncRNA = "/mnt/ls1/data/CPPred_data/Human/Homo38.small_ncrna_test.fa"
test_2_pcts = "/mnt/ls1/data/CPPred_data/Human/Human.small_coding_RNA_test.fa"
test_3_lncRNA = "/mnt/ls1/data/CPPred_data/Mouse/Mouse_ncrna.fa"
test_3_pcts = "/mnt/ls1/data/CPPred_data/Mouse/Mouse_coding_RNA.fa"
test_4_lncRNA = "/mnt/ls1/data/CPPred_data/Mouse/Mouse_small_ncrna.fa"
test_4_pcts = "/mnt/ls1/data/CPPred_data/Mouse/Mouse_small_coding_RNA.fa"

def model_performance(predicted_probas,y_pred,y_test):  
	cm = confusion_matrix(y_test, y_pred)
	specificity = cm[0][0]/float(cm[0][0]+cm[0][1])
	fpr, tpr, auc_thresholds = roc_curve(y_test, predicted_probas )
	auc_final = auc(fpr, tpr)
	accuracy = accuracy_score(y_test, y_pred)
	mcc = matthews_corrcoef(y_test, y_pred)
	[precision, recall, fbeta_score, support ] = precision_recall_fscore_support(y_test, y_pred)
	metrics = [specificity, recall[1], precision[1], accuracy, fbeta_score[1],auc_final, mcc]
	return metrics

def coding_nocoding_potential(input_file):
	coding={}
	noncoding={}
	for line in open(input_file).readlines():
		fields = line.split()
		if fields[0] == 'hexamer':continue
		coding[fields[0]] = float(fields[1])
		noncoding[fields[0]] =  float(fields[2])
	return coding,noncoding

def output_feature(seq_file,species="Human"):
	hex_file = 'Human_Hexamer.tsv'
	coding,noncoding = coding_nocoding_potential(hex_file)
	feature = []     
	length = 0    
	for seq in Seq.parse(seq_file,'fasta'):
        #seq -> seq.seq
		A,T,G,C,AT,AG,AC,TG,TC,GC,A0,A1,A2,A3,A4,T0,T1,T2,T3,T4,G0,G1,G2,G3,G4,C0,C1,C2,C3,C4 = CTD.CTD(seq.seq)
		insta_fe,PI_fe,gra_fe = PP.param(seq.seq)
		fickett_fe = fickett.fickett_value(seq.seq)
		hexamer = FrameKmer.kmer_ratio(seq.seq,6,3,coding,noncoding)
		Len,Cov,inte_fe = ORF_len.len_cov(seq.seq)
		vector = [Len,Cov,inte_fe,hexamer,fickett_fe,insta_fe,PI_fe,gra_fe,A,T,G,C,AT,AG,AC,TG,TC,GC,A0,A1,A2,A3,A4,T0,T1,T2,T3,T4,G0,G1,G2,G3,G4,C0,C1,C2,C3,C4]# + psedncfea
		feature.append(vector)  
		length = length + 1        
	return feature,length         

def save_results(results_file,metrics,time):
    with open(results_file,"a+") as op:
        if os.path.getsize(results_file):
            op.write(str(metrics[0])+"\t"+str(metrics[1])+"\t"+str(metrics[2])+"\t"+str(metrics[3])+"\t"+str(metrics[4])+"\t"+str(metrics[5])+"\t"+str(metrics[6])+"\t"+str(time)+"\n")
        else:
            op.write("Type\tspecificity\trecall\tprecision\taccuracy\tF1\tauc\tmcc\ttime\n")
            op.write(str(metrics[0])+"\t"+str(metrics[1])+"\t"+str(metrics[2])+"\t"+str(metrics[3])+"\t"+str(metrics[4])+"\t"+str(metrics[5])+"\t"+str(metrics[6])+"\t"+str(time)+"\n")

def oversamp_data(X_trpo,X_trne):
    ORF_Len = 303
    idx_po = X_trpo[:,0]<ORF_Len
    X_po_new = X_trpo[idx_po,:]   
    np.random.seed(0)
    random.seed(0)
#    p  = 4    8    12    16  20    24    28      32    36    40  
#       1796 3592 5388 7184  8980 10776  12572 14368  16164 17960
    add_num = 7184#int(len(X_po_new)*p)#
    print(add_num)
    X_add_all = []
    for i in range(add_num):
        idx_ram = random.randint(0,X_po_new.shape[0]-1)
        X = X_po_new[idx_ram,:]
        X_isone = abs(X)==1
        value = np.random.rand(1,38)
        add_value = value*0.005*X*(~X_isone)
        add_value[0,0] = 0 # ORFLen not be added
        X_add = X + add_value
        X_add = np.squeeze(X_add)
        X_add_all.append(X_add)
    X_add_all = np.array(X_add_all)   
    X_trpo = np.concatenate((X_trpo, X_add_all))    
    return X_trpo,X_trne

#def oversamp_data_smo(X_trpo,X_trne):
#    ORF_Len = 303
#    idx_po = X_trpo[:,0]<ORF_Len
#    X_po_new = X_trpo[idx_po,:]
#    
#    p = 16#64#2 # 4  8  16  24  32  40
#    now_num = int(len(X_po_new)*(p + 1))
#    X_ne_new = X_trne[:now_num]
#    
#    X = np.concatenate((X_po_new , X_ne_new))
#    y = np.array([1]*len(X_po_new)+[0]*len(X_ne_new))    
#    
#    smo = RandomOverSampler(random_state=0)
##    smo = ADASYN()
##    smo = SMOTE()
#    X_smo, y_smo = smo.fit_sample(X, y)
#    num = int(len(X))
#    X_smo_new = X_smo[num:]
#    print(len(X_smo_new))    
#
#    X_trpo = np.concatenate((X_trpo, X_smo_new))
#
#    return X_trpo,X_trne

def train(train_pcts,train_lncRNA):
	X_pos,len1 = output_feature(train_pcts)
	X_neg,len2 = output_feature(train_lncRNA)	
#	y_train = np.array([1]*len1+[0]*len2)
	X_pos = np.array(X_pos)
	X_neg = np.array(X_neg)
	X_pos,X_neg = oversamp_data(X_pos,X_neg)
	y_train = np.array([1]*X_pos.shape[0] + [0]*X_neg.shape[0])
    
	X_train = np.concatenate((X_pos,X_neg),axis = 0)
	scaler = StandardScaler()
	X_train = scaler.fit(X_train).transform(X_train)
	mean = scaler.mean_
	std = np.sqrt(scaler.var_)

	clf = XGBClassifier(n_estimators=800, random_state=0)
	classifier = clf.fit(X_train, y_train)
	data = {'classifier':classifier, 'mean':mean, 'std':std}
	# save classifier
	file = open('classifier.pkl','wb')
	pickle.dump(data,file)
	file.close()

def test(test_pcts,test_lncRNA): 
	save_path = 'classifier.pkl'
	data = pickle.load(open(save_path, 'rb'))   
	classifier = data['classifier']
	mean = data['mean']
	std = data['std']
	start=time.time()    
	X_pos,len1 = output_feature(test_pcts)
	X_neg,len2 = output_feature(test_lncRNA)	
	y_test = np.array([1]*len1+[0]*len2)
	X_pos = np.array(X_pos)
	X_neg = np.array(X_neg)
	X_test = np.concatenate((X_pos,X_neg),axis = 0) 

	X_test = X_test - mean
	X_test = X_test/std

	y_pred = classifier.predict(X_test)
	predicted_probas = classifier.predict_proba(X_test)[:, 1]
	end=time.time()
    #time -> time_
	time_ = (end-start)   
	metrics = model_performance(predicted_probas,y_pred,y_test)
	# save  metrics,time
	results_file = 'result.txt'
	save_results(results_file,metrics,time_)  

if __name__ == '__main__':
    train(train_pcts,train_lncRNA)
    test(test_1_pcts,test_1_lncRNA)
    test(test_2_pcts,test_2_lncRNA)
    test(test_3_pcts,test_3_lncRNA)
    test(test_4_pcts,test_4_lncRNA)   
