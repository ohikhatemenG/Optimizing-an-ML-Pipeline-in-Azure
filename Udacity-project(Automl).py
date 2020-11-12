#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from azureml.core import Workspace, Experiment
ws = Workspace.from_config()

ws = Workspace.get(name='quick-starts-ws-125398',
                   subscription_id='ef22e2eb-a072-4db6-86df-7ea6643b1b5f',
                   resource_group='aml-quickstarts-125398',
                   )

experiment = Experiment(ws, 'myexperiment')


# In[ ]:


for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":",compute.type)


# In[ ]:


from azureml.data.dataset_factory import TabularDatasetFactory

# TOO Create TabularDataset using TabularDataFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

ds = TabularDatasetFactory.from_delimited_files(path=url)


# In[ ]:


from train22 import clean_data
from sklearn.model_selection import train_test_split

# Use the clean_data function to clean your data.
# x, y = clean_data(### YOUR DATA OBJECT HERE ###)
x, y = clean_data(ds) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
train_data = x_train.join(y_train)


# In[ ]:


from azureml.train.automl import AutoMLConfig

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.

automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric="accuracy",
    training_data=train_data,
    label_column_name='y',
    n_cross_validations=5)


# In[ ]:


# Submit your automl run

### YOUR CODE HERE ###
exp = Experiment(ws, 'automl')  
auto_run = exp.submit(automl_config, show_output = True)


# In[ ]:


# Retrieve and save your best automl model.
### YOUR CODE HERE ###
best_run, fitted_model = auto_run.get_output()
joblib.dump(fitted_model, filename='outputs/best-automl.joblib')


# In[ ]:


# register the best model
model = auto_run.register_model(model_name='best-automl')


# In[ ]:


# Delete cluster
if(cluster_type == 0):
    aml_compute.delete()
else:
    compute_target.delete()

