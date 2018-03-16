

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)
from sklearn.linear_model import LogisticRegression, Lasso, LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_score
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')


def get_clean_referrals():
    referrals = pd.read_csv('data/2017_refs.csv', sep='|')
    referrals.drop(referrals.index[len(referrals)-1], inplace=True)
    referrals.columns = [col.lower().replace(' ', '_') for col in referrals.columns]
    referrals['refstat'].replace(['APPROVED', 'REJECTED', 'OTHER'], [1,0,0], inplace=True)
    referrals['dater'] =  pd.to_datetime(referrals['dater'], infer_datetime_format=True)
    referrals['regdate'] =  pd.to_datetime(referrals['regdate'], infer_datetime_format=True)
    referrals['sex'].replace(['F', 'M', 'I'], [0,1,0], inplace=True)
    referrals.rename(index=str, columns={"sex": "is_male", "refstat": "is_approve"}, inplace=True)
    referrals.drop(['plantype', 'patient', 'regdate', 'is_male', 'age', 'created_by', 'site_name'], axis = 1, inplace=True)
    referrals['priority_'].fillna('UNK', inplace=True)
    referrals['ppl'].replace(['Y', 'N', 'P'], [1,0,0], inplace=True)
    referrals['pat_req'].fillna('Y', inplace=True)
    referrals['pat_req'].replace(['Y', 'N'], [1,0], inplace=True)
    return referrals





class CustomScaler(BaseEstimator,TransformerMixin):
    # note: returns the feature matrix with the binary columns ordered first
    def __init__(self,bin_vars_index,cont_vars_index,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.bin_vars_index = bin_vars_index
        self.cont_vars_index = cont_vars_index

    def fit(self, X, y=None):
        self.scaler.fit(X[:,self.cont_vars_index], y)
        return self

    def transform(self, X, y=None, copy=None):
        X_tail = self.scaler.transform(X[:,self.cont_vars_index],y,copy)
        return np.concatenate((X[:,self.bin_vars_index],X_tail), axis=1)





## create a new var that is a binned (by historical approval rates) version of another categorical variable
def get_binned_var(cat_var, target_var, df):
    df[cat_var].fillna('wasnull', inplace=True)
    temp_df = df.groupby(cat_var)[target_var].mean()
    temp_df = pd.pivot_table(df, values=target_var, index=cat_var, aggfunc='mean')
    df[(''.join([cat_var, 'hist']))] = df[cat_var].apply(lambda x: temp_df.loc[x])
    return df, temp_df





def get_binned_var_from_hist(df, cat_var, hist_lookup):
    wn = pd.Series(df['is_approve'].mean())
    wn.name = 'wasnull'
    hist_lookup.append(wn)
    df[cat_var].fillna('wasnull', inplace=True)
    df[(''.join([cat_var, 'hist']))] = df[cat_var].apply(lambda x: hist_lookup.loc[x][0] if x in hist_lookup.index else df['is_approve'].mean())
    return df





# get binned for ref_type, CPT (lowest of)
# get dummie for pat_req, priority_





def plotroc(FPR, TPR):
    roc_auc = auc(FPR, TPR)
    plt.figure()
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')





def plot_prec_aa(precs, aarates):
    plt.figure()
    lw = 2
    plt.plot(aarates, precs, color='red',
             lw=lw)
    plt.plot([0, 1], [0.98, 0.98], color='black', lw=lw, linestyle='--')
    plt.plot([.4, .4], [.9, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.xlabel('Auto Approval Rate')
    plt.ylabel('Precision')
    plt.text(0.6, .99, 'Goal', color='green')
    plt.title('Auto-Approval rate & Precision')
    plt.savefig('AA_prec.png')





lw = 2
plt.figure(figsize=(8,8)
plt.plot([0, 1], [0.98, 0.98], color='black', lw=lw, linestyle='--')
plt.plot([.4, .4], [.9, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.xlabel('Auto Approval Rate')
plt.ylabel('Precision')
plt.text(0.6, .99, 'Goal', color='green')
plt.title('Auto-Approval rate & Precision')
plt.savefig('AA_prec_goal.png')





def get_binned_multiple(cols_to_bin, target_var, df):
    historical_dfs = []
    for col in cols_to_bin:
        df, new_hist_df = get_binned_var(col, target_var, df)

        historical_dfs.append((''.join([col, 'histdf'])))
    return df, historical_dfs





def ROC_curve(probabilities, labels):
    prob_vals = pd.DataFrame()
    prob_vals = pd.DataFrame({'probabilities':probabilities, 'labels':labels})
    prob_vals.sort_values('probabilities', inplace=True)
    prob_vals['prediction']=1
    lst_thresh = []
    lst_TPR = []
    lst_FPR = []
    for i in np.arange(prob_vals['probabilities'].count()):
        threshold = prob_vals['probabilities'].iloc[i]
        prob_vals.set_value(i, col='prediction', value=0)
        TPR = (prob_vals[(prob_vals['prediction']==1)&(prob_vals['labels']==1)]['labels'].sum())/(prob_vals['labels'].sum() if prob_vals['labels'].sum() > 0 else 1)
        FPR = (prob_vals[(prob_vals['prediction']==1)&(prob_vals['labels']==0)]['prediction'].sum())/(prob_vals[prob_vals['labels']==0]['labels'].count())
        lst_thresh.append(threshold)
        lst_TPR.append(TPR)
        lst_FPR.append(FPR)
    return lst_TPR, lst_FPR, lst_thresh





def get_prec_aa_prof(thresholds, y_true, y_proba):
    precs = []
    aarate = []
    n = len(y_true)
    for thresh in thresholds:
        tp, fp, fn, tn = standard_confusion_matrix(y_true, (y_proba > thresh).astype(int)).ravel()
        precs.append(tp/(tp + fp))
        aarate.append(tp/n)
    return precs, aarate

##################################

referrals = get_clean_referrals()

CPT = pd.pivot_table(referrals, values='is_approve', index='cpt1', aggfunc={'count', 'mean', 'std'})
CPT.to_csv('cpt_app_rate.csv')
dol_bins = np.linspace(0,20000,41)

def get_app_rates_by_dollar(dollar_bins, referrals_df):
    rates = []
    for i in dollar_bins:
        rate = referrals[(referrals['total_cost']>i) & (referrals['total_cost']<(i+500))]['is_approve'].mean()
        rates.append(rate)
    return rates





# rates = get_app_rates_by_dollar(dol_bins, referrals)





fig1 = plt.figure(figsize=(7,4))
ax11 = fig1.add_subplot(111)
ax11.hist(pd.pivot_table(referrals, values='is_approve', index='ref_prov', aggfunc='mean'), bins = 20);





fig2 = plt.figure(figsize=(7,4))
ax21 = fig2.add_subplot(111)
ax21.hist(pd.pivot_table(referrals, values='referral_key', index='ref_to_prov', aggfunc='count'), bins=20);


# ## Train Test Split




## break data into train and test

test = referrals[referrals['dater']>'2017-08-31']
train = referrals[referrals['dater']<='2017-08-31']





## break train data into val and train_train
train_val = train[train['dater']>'2017-07-31']
train_train = train[train['dater']<='2017-07-31']





## create features based on train_train data

train_train, planname_hist = get_binned_var('planname', 'is_approve', train_train)





train_train, priority_hist = get_binned_var('priority_', 'is_approve', train_train)





train_train, ref_to_prov_hist = get_binned_var('ref_to_prov', 'is_approve', train_train)





train_train, ref_to_spec_hist = get_binned_var('ref_to_spec', 'is_approve', train_train)





train_train, ref_prov_hist = get_binned_var('ref_prov', 'is_approve', train_train)





train_train, ref_spec_hist = get_binned_var('ref_spec', 'is_approve', train_train)





train_train, ref_type_hist = get_binned_var('ref_type', 'is_approve', train_train)





train_train, diag_hist = get_binned_var('diag', 'is_approve', train_train)





train_train, cpt1hist = get_binned_var('cpt1', 'is_approve', train_train)

# train_train = get_binned_var_from_hist(train_train, 'cpt2', cpt1hist)
# train_train = get_binned_var_from_hist(train_train, 'cpt3', cpt1hist)
# train_train = get_binned_var_from_hist(train_train, 'cpt4', cpt1hist)





## apply historical features to validation data

train_val = get_binned_var_from_hist(train_val, 'planname', planname_hist)





train_val = get_binned_var_from_hist(train_val, 'priority_', priority_hist)





train_val = get_binned_var_from_hist(train_val, 'ref_to_prov', ref_to_prov_hist)





train_val = get_binned_var_from_hist(train_val, 'ref_to_spec', ref_to_spec_hist)





train_val = get_binned_var_from_hist(train_val, 'ref_prov', ref_prov_hist)





train_val = get_binned_var_from_hist(train_val, 'ref_spec', ref_spec_hist)





train_val = get_binned_var_from_hist(train_val, 'ref_type', ref_type_hist)





train_val = get_binned_var_from_hist(train_val, 'diag', diag_hist)





train_val = get_binned_var_from_hist(train_val, 'cpt1', cpt1hist)





## apply historical features to test data

test = get_binned_var_from_hist(test, 'planname', planname_hist)





test = get_binned_var_from_hist(test, 'priority_', priority_hist)





test = get_binned_var_from_hist(test, 'ref_to_prov', ref_to_prov_hist)





test = get_binned_var_from_hist(test, 'ref_to_spec', ref_to_spec_hist)





test = get_binned_var_from_hist(test, 'ref_prov', ref_prov_hist)





test = get_binned_var_from_hist(test, 'ref_spec', ref_spec_hist)





test = get_binned_var_from_hist(test, 'ref_type', ref_type_hist)





test = get_binned_var_from_hist(test, 'diag', diag_hist)





test = get_binned_var_from_hist(test, 'cpt1', cpt1hist)





sns.set(style="white")

# Compute the correlation matrix
corr = train_train.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


fig = sns_plot.get_figure()
fig.savefig("imgs/corrheat.png")





test.corr()





prplotimag = sns.pairplot(train_train[['diaghist', 'ref_to_spechist', 'cpt1hist', 'ref_provhist']][120000:120500])





fig4 = prplotimag.
fig4.savefig("imgs/corrheat.png")





prplotimag.get_figur





plt.hist(train_train['ref_spechist'], bins=25);





plt.hist(train_train['ref_to_spechist'], bins=25);





plt.hist(train_train['cpt1hist'], bins=25);





def standard_confusion_matrix(y_true, y_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])





test.info()





cols_all = ['plannamehist', 'priority_hist', 'ref_to_provhist', 'ref_to_spechist', 'ref_provhist', 'ref_spechist', 'ref_typehist', 'diaghist', 'cpt1hist']





cols_select = ['ref_to_provhist', 'ref_typehist', 'diaghist', 'cpt1hist']





cols_few1 = ['ref_to_provhist']





cols_few2 = ['ref_to_provhist', 'cpt1hist']





## Split train_train into y and x
y_train_train = train_train['is_approve']
x_train_train = train_train[cols_all]
x_train_train_few = train_train[cols_select]





## Split train_val into y and x
y_train_val = train_val['is_approve']
x_train_val = train_val[cols_all]
x_train_val_few = train_val[cols_select]





## Split test into y and x
y_test = test['is_approve']
x_test = test[cols_all]
x_test_few = test[cols_select]
x_test_few2 = test[cols_few2]





thresholds = np.linspace(0,1,101)





## All in - all vars, no balance
modellog_nobal = LogisticRegression()
modellog_nobal.fit(x_train_train, y_train_train)
y_train_val_proba_nobal = modellog_nobal.predict_proba(x_train_val)[:,1]
FPR_nobal, TPR_nobal, thresholds_nobal = roc_curve(y_train_val, y_train_val_proba_nobal)

plotroc(FPR_nobal, TPR_nobal)





precisionsnobal, aaratesnobal = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_nobal)
plot_prec_aa(precisionsnobal, aaratesnobal)





#All in - balanced - all vars, balanced
modellog = LogisticRegression(class_weight='balanced')
modellog.fit(x_train_train, y_train_train)
y_train_val_pred = modellog.predict(x_train_val)
y_train_val_proba = modellog.predict_proba(x_train_val)[:,1]
FPR, TPR, thresholds = roc_curve(y_train_val, y_train_val_proba)

plotroc(FPR, TPR)





modellog.coef_





modellog_nine.coef_





modellog_three.coef_





modellog_few2.intercept_





modellog_few2.coef_





#precisions, aarates = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba)
plot_prec_aa(precisions, aarates)





## Lasso in Logistic - C=0.9
modellog_nine = LogisticRegression(penalty='l1', C=.9)
modellog_nine.fit(x_train_train, y_train_train)
y_train_val_proba_nine = modellog_nine.predict_proba(x_train_val)[:,1]
FPR_nine, TPR_nine, thresholds_nine = roc_curve(y_train_val, y_train_val_proba_nine)

plotroc(FPR_nine, TPR_nine)





precisions_nine, aarates_nine = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_nine)
plot_prec_aa(precisions_nine, aarates_nine)





## Lasso in Logistic - C=0.3
# modellog_three = LogisticRegression(penalty='l1', C=.3)
# modellog_three.fit(x_train_train, y_train_train)
# y_train_val_proba_three = modellog_three.predict_proba(x_train_val)[:,1]
# FPR_three, TPR_three, thresholds_three = roc_curve(y_train_val, y_train_val_proba_three)

plotroc(FPR_three, TPR_three)





#precisions_three, aarates_three = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_three)
plot_prec_aa(precisions_three, aarates_three)





## Downsample the "approve" class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(x_train_train, y_train_train)





## Logistic with downsampled "approves" to train with
# modellog_ds = LogisticRegression()
# modellog_ds.fit(X_resampled, y_resampled)
# y_train_val_pred_ds = modellog_ds.predict(x_train_val)
# y_train_val_proba_ds = modellog_ds.predict_proba(x_train_val)[:,1]
# FPR_ds, TPR_ds, thresholds_ds = roc_curve(y_train_val, y_train_val_proba_ds)

plotroc(FPR_ds, TPR_ds)





#precisions_ds, aarates_ds = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_ds)
plot_prec_aa(precisions_ds, aarates_ds)





# Logistic with selected variables - 'ref_to_provhist', 'ref_typehist', 'diaghist', 'cpt1hist'
# modellog_few = LogisticRegression()
# modellog_few.fit(x_train_train_few, y_train_train)
# y_train_val_proba_few = modellog_few.predict_proba(x_train_val_few)[:,1]
# FPR_few, TPR_few, thresholds_few = roc_curve(y_train_val, y_train_val_proba_few)

plotroc(FPR_few, TPR_few)





#precisions_few, aarates_few = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_few)
plot_prec_aa(precisions_few, aarates_few)





cols_few1 = ['ref_to_provhist']
x_train_train_few1 = train_train[cols_few1]
x_train_val_few1 = train_val[cols_few1]

## Logistic with selected variables - ref_to_provhist
modellog_few1 = LogisticRegression()
modellog_few1.fit(x_train_train_few1, y_train_train)
y_train_val_proba_few1 = modellog_few1.predict_proba(x_train_val_few1)[:,1]
FPR_few1, TPR_few1, thresholds_few1 = roc_curve(y_train_val, y_train_val_proba_few1)

plotroc(FPR_few1, TPR_few1)





#precisions_few1, aarates_few1 = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_few1)
plot_prec_aa(precisions_few1, aarates_few1)





cols_few2 = ['ref_to_spechist', 'cpt1hist', 'ref_to_provhist', 'ref_provhist', 'pat_req']
x_train_train_few2 = train_train[cols_few2]
x_train_val_few2 = train_val[cols_few2]

## Logistic with selected variables - ref_to_provhist
modellog_few2 = LogisticRegression()
modellog_few2.fit(x_train_train_few2, y_train_train)
y_train_val_proba_few2 = modellog_few2.predict_proba(x_train_val_few2)[:,1]
FPR_few2, TPR_few2, thresholds_few2 = roc_curve(y_train_val, y_train_val_proba_few2)

plotroc(FPR_few2, TPR_few2)





precisions_few2, aarates_few2 = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_few2)





plot_prec_aa(precisions_few2, aarates_few2)





## Logistic with selected variables - ref_to_provhist, cpt1hist, no intercept
modellog_few2_ni = LogisticRegression(fit_intercept=False)
modellog_few2_ni.fit(x_train_train_few2, y_train_train)
y_train_val_proba_few2_ni = modellog_few2_ni.predict_proba(x_train_val_few2)[:,1]
FPR_few2_ni, TPR_few2_ni, thresholds_few2_ni = roc_curve(y_train_val, y_train_val_proba_few2_ni)

plotroc(FPR_few2_ni, TPR_few2_ni)





## Logistic with selected variables - ref_to_provhist, cpt1hist, no intercept
modellogcv_few2 = LogisticRegressionCV(scoring='precision_score')





modellogcv_few2.fit(x_train_train_few2, y_train_train)





y_train_val_proba_few2 = modellog_few2_ni.predict_proba(x_train_val_few2)[:,1]
FPR_few2, TPR_few2, thresholds_few2 = roc_curve(y_train_val, y_train_val_proba_few2)

plotroc(FPR_few2, TPR_few2)





cols_few5 = ['ref_to_provhist', 'cpt1hist', 'pat_req']
x_train_train_few5 = train_train[cols_few5]
x_train_val_few5 = train_val[cols_few5]

## Logistic with selected variables - ref_to_provhist
modellog_few5 = LogisticRegression()
modellog_few5.fit(x_train_train_few5, y_train_train)
y_train_val_proba_few5 = modellog_few5.predict_proba(x_train_val_few5)[:,1]
FPR_few5, TPR_few5, thresholds_few5 = roc_curve(y_train_val, y_train_val_proba_few5)

plotroc(FPR_few5, TPR_few5)





precisions_few5, aarates_few5 = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_few5)
plot_prec_aa(precisions_few5, aarates_few5)





## Logistic with selected variables - ref_to_provhist
# y_test_proba_few2 = modellog_few2.predict_proba(x_test_few2)[:,1]
# FPR_few2_test, TPR_few2_test, thresholds_few2_test = roc_curve(y_test, y_test_proba_few2)

plotroc(FPR_few2_test, TPR_few2_test)





#precisions_few2_test, aarates_few2_test = get_prec_aa_prof(thresholds, y_test, y_test_proba_few2)
plot_prec_aa(precisions_few2_test, aarates_few2_test)





precisions, aarates = get_prec_aa_prof(thresholds, y_train_val, y_train_val_proba_dsfew)





plot_prec_aa(precisions, aarates)
