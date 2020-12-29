# Heart Failure Prediction using Microsoft Azure

This is a Machine Learning Project which takes opensource dataset from Kaggle and trains it with 2 different algorithms.
First using AutoML and then using HyperDrive with tuned hyperparameters. The best model will later be deployed and tested.

## Dataset

### Overview
The data is taken from Kaggle repository. 

**Citation:** Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020)

Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Task
In this project, Azure AutoML and Hyperdrive will be used to make prediction on the death event based on patient's 12 clinical features.

**12 clinical features:**

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)

### Access
After downloading the Heart Failure Dataset from kaggle as a csv file, it is registered as a Dataset in the Azure Workspace in a Tabular form uploading from local system. 
It can be then accessed as **Dataset.get_by_name(ws, dataset_name)**

## Automated ML
The AutoML settings and configuration used are as follows:
```
automl_settings = {
    "experiment_timeout_minutes": 30,
    "max_concurrent_iterations": 4,
    "primary_metric" : 'accuracy',
    "n_cross_validations": 5
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=data_train,
                             label_column_name="DEATH_EVENT", 
                             enable_early_stopping= True,
                             featurization= 'auto',
                             **automl_settings
                            )
```
Experiment timeout is set to control the use of resources. Maximum 4 iterations can be run simultaneously to maximize usage. Classification task is performed as the target column DEATH_EVENT has binary (0,1) output with primary metric as Accuracy. Featurization is also done which automatically scales and normalizes the dataset. 

<img src="Screenshots/autoML_run.png">
<img src="Screenshots/automl_experiment.png">

### Results
The best performing model after training using AutoML is VotingEnsemble with the Accuracy of 88.49350649350649 %

The models that VotingEnsemble used with it's weight are:
```
'ensembled_algorithms': "['LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'LightGBM', 'RandomForest', 'ExtremeRandomTrees', 'GradientBoosting', 'RandomForest', 'XGBoostClassifier', 'LogisticRegression']"

'ensemble_weights': '[0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091]'
```

To improve the model we can use different target metric like AUC_weighted or Recall. We can also try hypertuning the parameters to see if there is any improvement.

<img src="Screenshots/automl_best_model.png">
<img src="Screenshots/automl_best_accuracy.png">

**Best Run Id**

<img src="Screenshots/automl_bestrunid.PNG">

**Best AutoML Model Registered**

<img src="Screenshots/automl_registered.PNG">

## Hyperparameter Tuning
As it is a binary classification problem, the model used for HyperDrive is Logistic Regression. 
- It is easy to understand
- Trains very easily and faster compared to complex models


The parameters used for hyperparameter tuning are:
- Regularization Strength (C) with range 0.1 to 1.0
    -- Inverse of regularization strength. Smaller values cause stronger regularization
- Max Iterations (max_iter) with values 50, 100, 150 and 200
    -- Maximum number of iterations to converge

<img src="Screenshots/hyperdrive_run.png">

### Results
The best Accuracy for the HyperDrive model is 7666666666666667 %
The best hyperparameters for this accuracy are:
- 'Regularization Strength:': 0.5077980350098886
- 'Max iterations:': 50

To improve the model we can use different target metric to get broader perspective. We can also try increasing the range of the hyperparameters to see if there is any improvement.
<img src="Screenshots/hyperdrive_bestmodel.png">

**Best HyperDrive Model Registered**

<img src="Screenshots/hyperdrive_registered.PNG">

## Model Deployment
The AutoMl model is deployed using Azure Container Instance as a WebService. Best run environment and score.py file is provided to the InferenceConfig.
Cpu_cores and memory_gb are initialized as 1 for the deployment configuration. The aci service is then created using workspace, aci service name, model, inference config and deployment configuration.

The model is successfully deployed as a web service and a REST endpoint is created with status Healthy. A scoring uri is also generated to test the endpoint.

<img src="Screenshots/model_deployment.png">

<img src="Screenshots/deployed_endpoint.png">

The endpoint is tested in 2 ways: 
- using endpoint.py file which passes 2 data points as json 
- using 3 random sample data points and to see the actual value and predicted value 

<img src="Screenshots/model_test.png">

## Screen Recording
Link to screencast: [Link](https://youtu.be/fj7Av9YiuiY)

## Future Improvements
- Larger dataset can be used to increase data quality
- Different models can also be used with hyperparameter tuning
- Feature engineering can be performed using PCA 
