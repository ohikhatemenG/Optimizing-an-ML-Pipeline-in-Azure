#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azureml.core import Workspace, Experiment
ws = Workspace.from_config()

ws = Workspace.get(name='quick-starts-ws-124493',
                   subscription_id='374bdf1a-c648-4244-a317-f0d1ef4b85c7',
                   resource_group='aml-quickstarts-124493',
                   )

experiment = Experiment(ws, 'myexperiment')


# In[2]:


from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# TOO Create compute cluster
cpu_cluster = "cpu-cluster"

# verify that the cluster is not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cpu_cluster)
    print('cluster exist')
except ComputeTargetException:
    compute_gg = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                          max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster, compute_gg)

    cpu_cluster.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)


# In[3]:


from azureml.train.hyperdrive import normal, uniform, choice
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.widgets import RunDetails
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
from azureml.train.sklearn import SKLearn
import os

# Specify parameter sampler
ps = RandomParameterSampling(
    {
      '--C': choice( 0.25, 0.5, 0.75, 1.0, 2),
      '--max_iter': choice(100, 150, 200, 250, 300)
  }
)


# In[4]:


# Specify policy
policy = BanditPolicy(
           slack_factor = 0.1,
           evaluation_interval = 2,
           delay_evaluation = 5)


# In[5]:


if "training" not in os.listdir():
    os.mkdir("./training")


# In[6]:


# Creat a SKLearn estimator for the use with train.py
est = SKLearn(source_directory = './',
                     entry_script = 'train22.py',
                     compute_target = cpu_cluster)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy
hyperdrive_config = HyperDriveConfig(estimator = est,
                             hyperparameter_sampling=ps,
                             policy=policy,
                             primary_metric_name="Accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=25,
                             max_concurrent_runs=4)


# In[7]:


# Submit your hyperparameter run to the experiment
hyperdrive_run = experiment.submit(hyperdrive_config)


# In[8]:


# show run details with the widgets
RunDetails(hyperdrive_run).show()
hyperdrive_run.wait_for_completion(show_output=True)


# In[9]:


best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()['runDefinition']['arguments']
print(parameter_values)


# In[11]:


import joblib
# Get your best run and save the model from that run.
print(best_run)
best_run_metrics=best_run.get_metrics()

# Get your best run id and accuracy
print("BEST RUN ID AND BEST RUN ACCURACY")
print("Best_run_id",best_run.id)
print("Best_run_accuracy",best_run_metrics['Accuracy'])


# In[ ]:




