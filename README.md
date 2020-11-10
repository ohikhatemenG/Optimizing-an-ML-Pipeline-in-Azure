 Optimizing an ML Pipeline in Azure
 
Overview

This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK 
and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

Summary

In this Project we engaged the use of UCI Bank Marketing Dataset, which is related with direct marketing campaigns of Portuguese banking 
institution.The method of marketing campaigns were conducted on phone calls.The outputs of the bank is for client to access bank term deposit.
In this project, we made use of two different approaches (Hyperdrive Tuning & Automl) to make prediction of the outputs variable(y) if the 
client will subscribe to bank term deposit would be 'yes' or 'no'. The best performance model was a VotingEnsemble from the AutoML run, 
achieving an accuracy of 0.9155.
The dataset is source from paper: S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. 
Decision Support Systems, Elsevier, 62:22-31, June 2014.

Scikit-Learn Pipeline

The first experiment is using hydrive to search for the best run metrics. The first step is to create a compute instance to run it.
From the instructions given, I chose "Standard_DV_V2" virtual machine with a maximum of a 4 nodes.
I then create a paremeter sampler and an early-stop policy. In the parameter sampler I specified the use of random sampling on the
Inverse of regularization strength (C) and Maximum number of iterations to coverge(max_iter) as given in the train script(train.py).




