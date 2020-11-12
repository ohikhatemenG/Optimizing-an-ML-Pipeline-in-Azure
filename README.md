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

The first experiment is using hydrive to search for the best run metrics. The first step I took was to create a compute instance to run it.
From the instructions given, I specified "Standard_DV_V2" virtual machine with a maximum of 4 nodes.
I then create a parameter sampler and an early-stop policy. In the parameter sampler I specified the use of random sampling on the
Inverse of regularization strength (C) and Maximum number of iterations to converge(max_iter) as given in the train script(train.py).
And for C I specified the choice distribution as (0.25, 0.5, 0.75, 1.0, 2) Regularization is used to prevent overfitting on the model,
smaller values specify stronger regularization. The best performance of the model that had C values is 0.75
The maximum number of iterations I specified a choice distribution in the set (100, 150, 200, 250, 300). This actions was taken for 
the solvers to converg.Theoretically this parameter could be use to avoid wasting time in models that will not converge, no matter
how long we let the solver run.

With respect to early stopping policy I used the Bandit Policy with slack_factor of 0.1, evaluation interval of 2, and delay evaluation of 5.
Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run.
This means that any training run with an accuracy 10% lower than the maximum reported at that interval is terminated without completing it.
The delay evaluation parameter means to avoids premature termination of training runs by allowing all configurations to run for a minimum
number of intervals.The reasons why I chosen random sampler is because of it support for discrete and continuous hyperparameters, and is support
early termination of low performance runs when compare to grid sampler that only support discrete hyperparameters. Also it is cheaper than grid
sampler and Bayesian sampler.One benefit of Bandit Policy is more aggressive in saving with a smaller allowable slack when compare to Median
stopping Policy

The next step is to create an estimator with the use of train22.py script, this script loads data from the internet and I clean the data with
a custom function, also split the dataset into train and test sets of 70% and 30% respectively with a sklearn function. And It is responsibility
of parsing of arguments selected by the sampler, defined the model to be trained and keep monitoring of the chosen metric.
I then constructed the configurations for the training runs using HyperDriveConfig. In this HyperDriveConfig I included the estimator previously
defined, add the sampling and early termination policy mentioned above, specified primary matric(accuracy) and its goal(maximum). Also I added 
maximum total runs of 25 and maximum concurrent number of runs as 4

After taking the above step, I submitted the experiment to be executed and called for RunDetails(experiment).show() so as to monitor it within the
notebook. Finally, when the results came out, i retrieved the best run parameters values and save it

AutoML

The first step I took was to load the dataset from the internet and applied the clean data customize function derived from train22.py script to clean 
the dataset. And split the dataset into train and test sets of 70% and 30% respectively with sklearn function. Then I constructed the configuration
for the training runs using AutoMLConfig. In AutoMLConfig I included task type (classification), iterations timeout minutes of 30, primary metric
(accuracy), cross validation of 5, compute target(y), training data (train data). Hyperdrive tuning was not needed in this AutoML.Finally, I 
submitted the runs, after the results came out, I had the best model as VotingEnsemble with accuracy of 0.9155. Then I retrieved the best model 
and saved it. AutoML detected a problem of imbalance data and suggested that the run cancel the current run and fix the balancing problem. Also no 
feature missing value was detected in the training data. The inputs were analyzed, and no high cardinality features were detected. The most influential
feature in the AutoML is duration.

Pipeline Comparison

The two experiments I ran gave quite similar accuracies, 91.03% for the logistic regression in the HyperDrive and 91.55% for the voting ensemble in the AutoML. 
However, once again these metrics might be misleading. If these results were correct, unless I am planning to deploy this models to classify a huge amount
of records, a 0.52% difference doesn´t seem relevant.

AutoML runs a relevant amount of its experiments when compared to the use of HyperDrive pipeline.
AutoML established a number of pipelines in parallel that make use of different algorithms and parameters for me
AutoML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is 
considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment.
HyperDrive pipeline process or procedure took a longer time when compared to AutoML Process.

Future work

(1) As mentioned above, the issue of imbalance data needs to be addressed. And one way to resolve it, is to use the SMOTE or ADASYN sampling methods in the
imbalanced-learn library.
(2) Remove the duration feature in the dataset if the model is to be used for prediction. First, this feature is unknown at the time of prediction. Second,
even if I knew it, it does not depend on anything the decision-makers can control, thus I don't see the use of it (except for academic exercises). One might 
say that age, for example, is not under the decision-makers control either; however, they can target their advertising campaign to specific age ranges, while
they cannot force potential customers to stay longer on the phone.









