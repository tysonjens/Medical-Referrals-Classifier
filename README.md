## Referrals Approval Classifier

A medical provider group reviews and approves or denies over 2 million referrals each year. It is costly to review each referral, so the group "auto-approves" 30% of all referrals from certain specialties when specific care is requested. There is appetite to increase the auto-approval rate to 40-50%, but approving a referral that should be denied is costly.

**Question:** Can a predictive model increase the auto-approval rate while keeping 'false positives' low?


## Table of Contents
1. [Measure of Success](#measure-of-success)
2. [Dataset](#dataset)
3. [Feature Engineering](#feature-engineering)
4. [Training, Validation & Test
Sets](#training,-validation-&-test-sets)
5. [Logistic Regression Models](#logistic-regression-models)
    * [All in](#all-in)
    * [All in - Balanced](#all-in---balanced)  
    * [Select](#select)
    * [Select - Downsampled](#select---downsampled)
    * [Results](#results)
6. [Future Directions](#future-directions)

## Measures of Success

1. Precision at 40% auto approval rate
2. Area under the receiver operation characteristc curve (AUC-ROC)

<img alt="Example of tumor segmentation overlay on T2" src="imgs/AA_prec_goal.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

## Dataset

Over two million referrals placed by physicians for their patients to see a specialist during the 2017 calendar year.

####Data Dictionary
Name | Variable Name | Description | Type
------|----------|--------|----
**Approve** (target) | refstat | 1 if referral approved, else 0 | bin
Date Received | dater | time / date stamp of when referral received | date
Registration Date | regdate | Date when the member registered with plan | date
Sex | sex | 1 if male, else 0 | bin
Age | age | integer age of patient | int
Priority | priority_ | Physicians can indicate "Routine", "urgent", "emergency" | cat (4)
Patient Request | pat_req | 1 if patient requested the referral, else 0 | bin
Referring Physician | ref_prov | name of physician submitting the referral | cat (4000)
Refer "To" Physician | ref_to_prov | name of physician received the referral | cat (10000)
Specialty | ref_to_spec | E.g. "Cardiology", "Dermatology" | cat (50)
Procedure Code | cpt1, cpt2 ... | What is being requested in the referral | cat (14000)

*HIPAA Note: all personal information was scrubbed from the data prior to use.  Age and sex are available for each referral, but the data contain no keys to tie referrals to patients.*

#### Correlations Between Target and Predictors
<img alt="Example of tumor segmentation overlay on T2" src="imgs/corrheat.png" width='500'>

<sub><b>Figure: </b> Correlations between approvals and predictors. </sub>

#### Feature Engineering

* Binary variables were updates to "1" and "0", and renamed as appropriate
* Categorical predictors were translated to continuous variables through the following steps:
  * Using training data, historical averages of the target variable were calculated *for each level*. For example, in the ref_to_spec column, "Cardiology" is one of 50 levels and historically approve at 96%.
  * In a new column, the historical averages are transcribed for each level.

It is important to engineer the features from "historical" data in order for the classifier to be viable.

#### Training, Test, and Validation Sets

The purpose of the model is to predict whether future approvals with auto-approval or not.  As such,
* test data are from the final 3 months of 2017
* valdiation data are from August, 2017
* training data are from the first 8 months of 2017


## Logistic Regression Models

#### Model 1 - Referring Provider Only


<img alt="Example of tumor segmentation overlay on T2" src="imgs/ROC_ few1.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>


<img alt="Example of tumor segmentation overlay on T2" src="imgs/AA_prec_few1.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

#### Model 2 - Referring Provider & CPT code

<img alt="Example of tumor segmentation overlay on T2" src="imgs/ROC_few2.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>


<img alt="Example of tumor segmentation overlay on T2" src="imgs/AA_prec_few2.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

#### Model 3 - Logistic w/ all vars w/ Penalty (C = 0.3)

<img alt="Example of tumor segmentation overlay on T2" src="imgs/AA_prec_loglas3.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

#### Model 4 - All variables, y-undersampled

<img alt="Example of tumor segmentation overlay on T2" src="imgs/AA_prec_ds.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

#### Model 2 on Test Data

<img alt="Example of tumor segmentation overlay on T2" src="imgs/ROC_test_few2.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>


<img alt="Example of tumor segmentation overlay on T2" src="imgs/AA_prec_test_few2.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

## Results

Num | Model | ROC-AUC | Precision at 40% AA
---|----|-----|----
1 | Logistic - Referring Provider | 0.76 | 97.1%
2 |Logistic - Refer To Provider, CPT1 | 0.79 | 97.7%
3 | Logistic w/ penalty (lasso, C=.3), all vars | 0.79 | 97.5%
4 |Logistic, all vars, y-undersampled | 0.79 | 97.6%
 - | - | -
Test | Logistic - Refer To Provider, CPT1| 0.78 | 97.6%

## Future Directions  
