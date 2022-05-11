# Predicting software fault types using bug reports
Debugging is a time-consuming and expensive process.
Developers have to select appropriate tools, methods and approaches in order to efficiently reproduce, localize and fix bugs.
These choices are based on the developers' assessment of the type of fault for a given bug report.
We propose a machine learning (ML) based approach that predicts the fault type for a given textual bug report.
We built a dataset from 70+ projects for training and evaluation of our approach.
Further, we performed a user study to establish a baseline for non-expert human performance on this task.
Our models, incorporating our custom preprocessing approaches, reach up to 0.69% macro average F1 score on this bug classification problem.
We demonstrate inter-project transferability of our approach.
We identify and discuss issues and limitations of ML classification approaches applied on textual bug reports.
Our models can support researchers in data collection efforts, as for example bug benchmark creation.
In future, such models could aid inexperienced developers in debugging tool selection, helping save time and resources.
More information can be found in the publication:
> Thomas Hirsch and Birgit Hofer: "Using textual bug reports to predict the fault category of software bugs", submitted to the Array Journal, under review.

# Structure of this repository
## `artifact_detection_model`
The natural language / non-natural lanugage classification model used for preprocessing bug report texts. 
Due to GitHub file size restrictions, the pretrained model used in our experiments is not included, please see [here](#artifact_detection-model) on how to obtain it.

## `dataset`
The datasets, survey results, and validation datasets are located in [dataset](dataset).
The full sample of 11621 bug reports mined from github can be found in [dataset/github_issues_dataset.csv.zip](dataset/github_issues_dataset.csv.zip).
The training set of 496 bug reports can be found in [dataset/classification_dataset.csv.zip](dataset/classification_dataset.csv.zip).

Evaluation of interrater agreements and survey scores is performed by the scripts located in [dataset/evaluation](dataset/evaluation), its outputs can be found in [dataset/evaluation/output](dataset/evaluation/output).

The final labels used in the experiments (Researcher 1) are located in [dataset/manualClassification](dataset/manualClassification).
The data for our internal and external validation steps can be found in [dataset/manualClassification/validation](dataset/manualClassification/validation).
Survey submissions are located in [dataset/survey/survey.csv](dataset/survey/survey.csv).
Results of our keyword search are located in [dataset/keyword_search](dataset/keyword_search).

## `classification`
### Single classifier models (RQ2 - EXP1 & EXP2)
The code of EXP1 can be found in [step_1_0_trivial_approach.py](classification/step_1_0_trivial_approach.py) and [step_1_1_trivial_approach_with_artifact_replacement.py](classification/step_1_1_trivial_approach_with_artifact_replacement.py).
Analog, code of EXP2 is located in [step_2_0_ncv_multiple_algorithms.py](classification/step_2_0_ncv_multiple_algorithms.py).

### Ensemble classifier model (RQ2 - EXP3)
The first step of nested cross validation for weighted class specific models is [step_3_ncv_multiple_algorithms_weighted_classes.py](classification/step_3_ncv_multiple_algorithms_weighted_classes.py).
The second step, ensembling these models and evaluation of these ensembles is located in [step_4_ensembles.py](classification/step_4_ensembles.py).

### Evaluation and plotting (RQ2)
The remaining scripts in [classification](classification) (step_Z_*.py) perform the final evaluation of the recorded scores, plotting, and further analysis.

### RQ2 Results
Outputs of each of the above steps, including plots, dataframes of scores, can be found in [classification/output](classification/output).

### Cross project transferability experiment (RQ3)
[classification/cross_project](classification/cross_project) contains all code for RQ3 experiments.
Ensemble classifiers are evaluated in [cross_project_evaluation.py](classification/cross_project/cross_project_evaluation.py), single classifier models in [cross_project_evaluation_LRC_SVC_RFC.py](classification/cross_project/cross_project_evaluation_LRC_SVC_RFC.py).

### Evaluation and plotting (RQ3)
The remaining scripts in [classification/cross_project](classification/cross_project) (cp_Z_*.py) perform the final evaluation of the recorded scores, plotting, and further analysis.

### RQ3 Results
Outputs of each of the above steps, including plots, dataframes of scores, can be found in [classification/cross_project/output](classification/cross_project/output).

## `preprocessing`
NLP preprocessing steps used in our experiments, such as de-camel-casing, and stemming are located in [preprocessing](preprocessing).

## `trainingsetCreation`
[trainingsetCreation](trainingsetCreation) contains the keyword search used to sample from the 11621 bug reports mined from GitHub, and to created the final training set [dataset/classification_dataset.csv.zip](dataset/classification_dataset.csv.zip) used in all our experiments.
The keywords used in this search are stored in [trainingsetCreation/keywordSearch/constants.py](trainingsetCreation/keywordSearch/constants.py).

# Usage
##Python requirements
Python 3.5+
* pandas
* scikit-learn
* matplotlib
* nltk

### Conda
A conda file for importing/creating an environment is defined in [environment.yml](environment.yml)
The exact conda environment used in our experiments is defined in [environment-full-versions.yml](environment-full-versions.yml).

## artifact_detection model
The models in this repository utilize a pretrained scikitlearn model to distinguish natural language from non-natural language portions in bug reports.
Due to filesize restrictions on GitHub, this pretrained model is not included, however there are two options to obtain it:
1. Download the pretrained model [here](https://drive.google.com/file/d/1elXrmciHrUuN9_iQYPcnRXx18ZlqMqzt/view?usp=sharing) and move the file to [artifact_detection_model/out/](artifact_detection_model/out/)
2. Train the model yourself.
   - Clone [https://github.com/AmadeusBugProject/artifact_detection/releases/tag/v1.1](https://github.com/AmadeusBugProject/artifact_detection/releases/tag/v1.1)
   - Execute [https://github.com/AmadeusBugProject/artifact_detection/blob/master/artifact_detection_model/RUN_train_model.py](RUN_train_model.py) (approx. 5min runtime on older Intel i5 processors)
   - Move the [artifact_detection_model/out/artifact_detection.joblib](artifact_detection_model/out/artifact_detection.joblib) file into the same path in this project.

## Training set
To load the training set of 496 labeled bug reports:
```python
from classification.utils import load_dataset
docs, target, target_names = load_dataset()
```

## Classifiers
The default hyperparameters, the considered hyperparameter spaces, classifiers, and the default pipelines used in all our experiments are defined in [classification/defaults.py](classification/defaults.py).
To instantiate such classifiers:
```python
from classification.defaults import make_default_pipeline, make_default_gridsearch_parameters
from classification.utils import get_classifier_by_name
gs_params = make_default_gridsearch_parameters()
pipeline = make_default_pipeline(get_classifier_by_name('LinearSVC(random_state=42)'))
```

## Ensembling example
In [ensemble_example.py](ensemble_example.py), an ensemble classifier is created analog to our RQ3 experiments, but splitting training and test sets randomly.

First, loading the dataset and creating test/training splits:
```python
docs, target, target_names = load_dataset()
train_docs, validation_docs, train_target, validation_target = train_test_split(docs, target, test_size=0.2, random_state=42)
```
Then running nested grid search for all categories and classifier algorithms:
```python
nested_cross_val_classifiers = nested_cross_validation_for_model_selection(train_docs, train_target, target_names)
```
And finally ensembling the top classifiers and evaluating this ensemble using the above generated validation set:
```python
num_classifiers_per_cat = 1
ensembe_performance = ensemble_best_classifiers(nested_cross_val_classifiers, num_classifiers_per_cat, train_docs, train_target, validation_docs, validation_target, target_names)
```

# Classification schema
Detailed documentation and examples for our fault type classification schema can be found in [classification_schema.md](classification_schema.md).

# Licence
The work contained in this repo is licenced under [LICENSE](AGPLv3), the bug reports contained in the datasets are subject to licence of their origin projects, please see [dataset/licences](dataset/licences) or the corresponding projects GitHub repositories.

# Acknowledgment
The work has been funded by the Austrian Science Fund (FWF): P 32653 (Automated Debugging in Use).
