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
And for C I specified the choice distribution as (0.25, 0.5, 0.75, 1.0, 2) Regularization is use to prevent overfitting on the model,
smaller values specify stronger regularization. The best performance of the model that had C values is 0.75
The maximum number of iterations I specified a choice distribution in the set (100, 150, 200, 250, 300). This actions was taken for 
the solvers to converg.Theoretically this parameter could be use to avoid waisting time in models that will not converge, no matter
how long we let the solver run.

With respect to early stopping policy I used the BanditPolicy with slack_factor of 0.1, evaluation interval of 2, and delay evaluation of 5.
Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.
This means that any training run with an accuracy 10% lower than the maximum reported at that interval is terminated without completing it.
The delay evaluation parameter means to avoids premature termination of training runs by allowing all configurations to run for a minimum
number of intervals.
The next step is to create an estimator with the use of train22.py script, this script load data from the internet and I clean the data with
a custom function, also split the dataset into train and test sets of 70% and 30% respectively with a sklearn function. And It is responsibility
of parsing of arguments selected by the sampler, defined the model to be trained and keep monitoring of the chosen metric.
I then constructed the configurations for the training runs using HyperDriveConfig. In this HyperDriveConfig I included the estimator previously
defined, add the sampling and early termination policy mentioned above, specified primary matric(accuracy) and its goal(maximum).







