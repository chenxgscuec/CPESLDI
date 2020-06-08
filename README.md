About
===============================
An implementation of CPE-SLDI, a new model for predicting coding potential.
In this work, we introduce oversampling technology and increase the number of positive instances and alleviate the problem of local data imbalance, and propose a coding potential prediction method, CPE-SLDI, which constructs a prediction model by combining various sequence-derived features based on the augmented data.


Requirements
========================
    [python 2.7](https://www.python.org/downloads/)


Data
=====================
In this work, we use the datasets organized by CPPred.
Human-Training: coding RNAs(33360,117M)+ noncoding RNAs(24163,24M)
Human-Testing: 8557 coding RNAs+ 8241 noncoding RNAs
Human-sORF-Testing: 641 coding RNAs+ 641 noncoding RNAs
Mouse-Testing: 31102 coding RNAs+ 19930 noncoding RNAs
Mouse-sORF-Testing: 846 coding RNAs+ 1000 noncoding RNAs
The data can be downloaded at http://www.rnabinding.com/CPPred/


Usages
========================
python CPESLDI.py


Reference
========================
If you find the code useful, please cite our paper.


Our manuscipt titled with "Predicting Coding Potential of RNA Sequences by Solving Local Data Imbalance" is being reviewed by IEEE/ACM Transactions on Computational Biology and Bioinformatics.



Contact
=====================
Author: Xian-gan Chen
Mail: chenxg@mail.scuec.edu.cn
Date: 2020-6-10
School of Biomedical Engineering, South-Central University for Nationalities, China