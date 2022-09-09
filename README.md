# Regression Project

## Project Objective 
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 

* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  

* Utilize charts and statistical tests

> Construct models to predict assessed home value for single family properties using regression techniques

> Make recommendations to a *fictional* data science team about how to improve predictions

> Refine work into report in form of jupyter notebook. 

> Present walkthrough of report in 5 minute presentation to classmates and instructors
* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Regression model that accurately predicts property tax assessed values (`taxvaluedollarcnt` - rename) of *Single Family Properties* using attributes of the properties? </br>

>Find key drivers of property value for single familty properties</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting these homes' values

## Deliverables
> Github repo with a complete readme.md, a final report (.ipynb), acquire & prepare modules (.py), other supplemental artifacts created while working on the project (e.g. exploratory/modeling notebook(s))</br>
> 5 minute presentation of your final notebook</br>

## Data Dictionary
|Target|Datatype|Definition|
|:-----|:-----|:-----|
|taxvaluedollarcnt|xxx non-null: uint8| property tax assessed values

|Feature|Datatype|Definition|
|:-----|:-----|:-----|
customer_id                           | 4225 non-null   object | customer company identification code
gender                                | 4225 non-null   object | customer gender 
senior_citizen                        | 4225 non-null   int64  | customer status as senior citizen
partner                               | 4225 non-null   object | customer partner status
dependents                            | 4225 non-null   object | customer dependent status
tenure                                | 4225 non-null   int64  | customer tenure in months
phone_service                         | 4225 non-null   object | customer phone service status
multiple_lines                        | 4225 non-null   object | customer subscription to multiple phone lines
online_security                       | 4225 non-null   object | customer online security service status
online_backup                         | 4225 non-null   object | customer online backup service status
device_protection                     | 4225 non-null   object | customer device protection service status
tech_support                          | 4225 non-null   object | customer tech support service status
streaming_tv                          | 4225 non-null   object | customer streaming tv service status
streaming_movies                      | 4225 non-null   object | customer streaming movie service status
paperless_billing                     | 4225 non-null   object | customer paperless billing status
monthly_charges                       | 4225 non-null   float64| customer monthly charges
total_charges                         | 4216 non-null   float64| customer total charges
churn                                 | 4225 non-null   object | customer churn status
contract_type                         | 4225 non-null   object | customer contract type
payment_type                          | 4225 non-null   object | customer payment type
internet_service_type                 | 4225 non-null   object | customer internet service 


## Initial Questions and Hypotheses
> Why do some properties have a much higher value than others when they are located so close to each other? 
* ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

* Conclusion: 

> Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location?
* ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

 * Conclusion: 

 >Is having 1 bathroom worse than having 2 bedrooms?
* ${\alpha}$ = .05

* ${H_0}$: 

* ${H_a}$: 

 * Conclusion: 

## Summary of Key Findings
* 
* 
* 
* 

## Pipeline Walkthrough
### Plan
> Create and build out project README
> Create required as well as supporting project modules and notebooks
* `env.py`, `wrangle.py`,  `model.py`,  `Final Report.ipynb`
* `wrangle.ipynb`,`model.ipynb` ,
* Handle acquire, explore, and scaling in wrangle


### Acquire
> Acquired zillow 2017 data from appropriate sources
* Create local .csv of raw data upon initial acquisition for later use
* Take care of any null values -> Decide on impute or elimination
> Add appropriate artifacts into `wrangle.py`

### Prepare
> Univariate exploration: 
* Basic histograms/boxplot for categories
> Took care of outliers
> Handle any possible threats of data leakage
> Create dummy vars
> Split data
> Scale data
> Collect and collate section *Takeaways*
> Add appropirate artifacts into `wrangle.py`

### Explore
> Bivariate exploration
* Investigate and visualize *all* features against home value
> Identify possible areas for feature engineering
* 
> Multivariate:
* Visuals exploring features as they relate to home value
> Statistical Analysis:
* Answer questions from *Initial Questions and Hyptheses* 
* Answer questions from *Univariate* and *Bivariate* exploration
> Collect and collate section *Takeaways*

### Model
> Ensure all data is scaled
> Set up comparison dataframes for evaluation metrics and model descriptions  
> Set Baseline Prediction and evaluate accuracy  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only `area`, `bedrooms`, `bathrooms` as features
> Ordinary Least Squares
* 
> LASSO + LARS
* 
> Polynomial Regression
* 
> Generalized Linear Model
* 

> Choose **three** models to validate
* 

>Choose **one** model to test
* 
> Collect and collate section *Takeaways*

### Deliver
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
* Verify docstring is implemented for each function within all notebooks and modules 
> Present to audience of CodeUp instructors and classmates
* 

## Project Reproduction Requirements
> Requires personal `env.py` file containing database credentials  
> Steps:
* Fully examine this `README.md`
* Download `acquire.py, model.py, prepary.py, and final_report.ipynb` to working directory
* Create and add personal `env.py` file to directory. Requires user, password, and host variables
* Run ``final_report.ipynb`