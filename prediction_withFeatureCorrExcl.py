' Script to predict LongCOVID based on proteomic and clinical data using a simple RF. Created by S Brueningk 2023 '

import pandas as pd
import shap
import numpy as np
from tqdm import tqdm
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  roc_auc_score, accuracy_score, \
balanced_accuracy_score, f1_score, recall_score, precision_score
import os
from pathlib import Path
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc,\
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


###########################################################################################################
# Functions
def createSplit(Y, metadata,n_splits = 5, stratifyFor = ['CareSamp']):

    if 'age_group' in stratifyFor:
        age_groups_max = [60, 150]
        age_group = []
        for s in metadata.index:
            this_age = metadata.loc[s, 'Age']
            age_group.append(str(age_groups_max[np.where(np.array(age_groups_max) >= this_age)[0][0]]))
        metadata['age_group'] = age_group


    # Check this - your variables need to be str here
    concat_labels = Y.copy().astype(int).astype(str)
    for i, var in enumerate(stratifyFor):
        for c in concat_labels.index:
            concat_labels.loc[c]= concat_labels.loc[c] + metadata[var].astype(str).loc[c]

    # Check label count for split:
    while np.any(np.array(list(Counter(concat_labels).values()))<4):
        keys = np.array(list(Counter(concat_labels).keys()))
        key_small_count = keys[np.array(list(Counter(concat_labels).values()))<4][0]

        # Change age group
        if '60' in key_small_count:
            this_agegroup = '60'
            this_agegroup_new = '150'
        else:
            this_agegroup = '150'
            this_agegroup_new = '60'

        new_label = key_small_count.split(this_agegroup)[0] +\
                    this_agegroup_new+\
                    key_small_count.split( this_agegroup )[1]


        inds_smallCount = concat_labels[concat_labels.values==key_small_count].index
        concat_labels.loc[inds_smallCount]=new_label


    # Perform multiple splits - train, val, test
    df = pd.Series(metadata.index)
    skf = StratifiedKFold(n_splits=n_splits)
    split = 0
    partition = dict()
    for train_index, test_index in skf.split(df, concat_labels):

        # First get validation and training together, then split again
        id_rest = df.loc[df.index.intersection(train_index)].values
        y_rest = concat_labels.iloc[train_index]
        id_trn, id_val, y_trn, y_val = train_test_split(id_rest, y_rest, test_size=0.25, random_state=1,
                                                        stratify=y_rest)

        # Test set
        id_tst = df.loc[df.index.intersection(test_index)].values

        # Save as dictionary
        partition['train'+str(split)] = list(id_trn)
        partition['validation'+str(split)] = list(id_val)
        partition['test'+str(split)] = id_tst
        split = split + 1

    try:
        Y = Y.to_frame()
    except:
        print('already DF')
    labels = dict(zip(Y.index.values, Y.astype(int)[Y.columns[0]].values))

    return partition, labels
def getData(partition, X, cv):
    X_train = X.loc[partition['train' + str(cv)], :]
    X_val = X.loc[partition['validation' + str(cv)], :]
    X_test = X.loc[partition['test' + str(cv)], :]

    return X_train, X_test,X_val
def getLabels(partition, labels, cv):
    # Generate data
    y_trn = [labels[ID] for i, ID in enumerate(partition['train' + str(cv)])]
    y_val = [labels[ID] for i, ID in enumerate(partition['validation' + str(cv)])]
    y_tst = [labels[ID] for i, ID in enumerate(partition['test' + str(cv)])]
    return y_trn, y_tst, y_val
def perf_90recall(models, X_tests, y_tests):
    if len(np.unique(y_tests)) == 2:

        pred_test = []
        y_test_merged = y_tests  # list(chain(*y_tests))

        pred_test = models.predict_proba(X_tests)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_test_merged, pred_test)
        threshold = thresholds[np.abs(recalls - 0.9).argmin()]

        pred_test = (models.predict_proba(X_tests)[:, 1] >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_merged, pred_test).ravel()
        performance = {}
        performance['precision'] = tp / (tp + fp)
        performance['specificity'] = tn / (tn + fp)
        performance['sensitivity'] = tp / (tp + fn)
        performance['f1'] = f1_score(y_test_merged, pred_test)
        performance['npv'] = tn / (tn + fn)
        performance['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
        performance['recall'] = tp / (tp + fn)
        # performance = pd.DataFrame(performance, index=[name])

        y_score = models.predict_proba(X_tests)[:, 1]
        aps = average_precision_score(y_test_merged, y_score)
        fp_rates, tp_rates, _ = roc_curve(y_test_merged, y_score)
        roc_auc = auc(fp_rates, tp_rates)

        # if not os.path.exists('{}/perf_90recall.csv'.format(output_path)):
        # performance.to_csv('{}/perf_90recall.csv'.format(output_path))
    # else:
    # performance.to_csv('{}/perf_90recall.csv'.format(output_path), header=False, mode='a')

    else:
        print('multi class!')
        # multiclass:
        # Macro averaged precision: calculate precision for all classes individually and then average them
        # Micro averaged precision: calculate class wise true positive and false positive and then use that to calculate overall precision

        tn = np.nan
        fp = np.nan
        fn = np.nan
        tp = np.nan
        y_pred = models.predict_proba(X_tests)
        y_pred_class = models.predict(X_tests)

        precision = precision_score(y_tests, y_pred_class, average='weighted')
        recall = recall_score(y_tests, y_pred_class, average='weighted')
        tp = accuracy_score(y_tests, y_pred_class)
        acc = balanced_accuracy_score(y_tests, y_pred_class)
        F1 = f1_score(y_tests, y_pred_class, average='weighted')
        roc_auc = roc_auc_score(y_tests, y_pred, average='weighted', multi_class='ovo')
        aps = np.nan  # average_precision_score(y, y_pred, average = 'weighted')

        performance = {}
        performance['precision'] = precision
        performance['f1'] = F1
        performance['accuracy'] = acc
        performance['recall'] = recall

    return tn, fp, fn, tp, performance['accuracy'], performance['precision'], \
           performance['recall'], roc_auc, aps, performance['f1']
def getPredictionSingleFold_nooutput(cv, partition, labels, clf_choice, clinical):

    # Get the labels
    y_train, y_test, y_val = getLabels(partition, labels, cv)
    y_trainVal = y_train+ y_val

    # Get data
    X_train, X_test, X_val = getData(partition, clinical, cv)
    X_trainVal = pd.concat([X_train,X_val])


    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in X_train.index else 0 for x in X_trainVal.index]
    ps = PredefinedSplit(test_fold=split_index)

    if clf_choice == 'MLP':
        clf = MLPClassifier(max_iter=1000)

        param_grid = {'solver': ['adam'],
                      'alpha': [0, 1e-04, 1e-02],
                      'activation': ['relu', 'logistic'],
                      'hidden_layer_sizes': [5, 10, 50],
                      'learning_rate_init': [0.001, 0.0001],
                      'early_stopping': [True],
                      'validation_fraction': [0.25]}
        clf = GridSearchCV(clf, param_grid, cv=5, verbose=1, scoring='roc_auc',refit=True)

    if clf_choice == 'LR':

        # Initialise logistic regression model, optimise hyperparameters by gridsearch
        clf = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=1)
        param_grid = {'penalty': ["l1", 'l2', 'elasticnet'],
                      'C': np.logspace(-7, 4, 12),
                      'solver': ['liblinear']}
        clf = GridSearchCV(clf, param_grid, cv=5, verbose=1, scoring='roc_auc',refit=True)

    if clf_choice == 'RF':
        # Initialise random forest, optimise hyperparameters by random grid
        clf = RandomForestClassifier(class_weight='balanced', random_state=1)

        # Number of trees in random forest
        n_estimators = [50, 200, 1000]

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree
        max_depth = [3,5,7]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [3,5,10]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [3, 6, 9]

        # Method of selecting samples for training each tree
        bootstrap = [True]

        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap,
                      'class_weight': ['balanced']}
        # clf = RandomizedSearchCV(clf, param_grid, cv=5, n_iter=50, verbose=1, scoring='roc_auc')
        clf = GridSearchCV(clf, param_grid, cv=5, verbose=1, scoring='roc_auc',refit=True)

    # Fit
    clf.fit(X_trainVal, y_trainVal)

    # Save best estimator
    best_est = clf.best_estimator_
    best_est.fit(X_trainVal, y_trainVal)

    return best_est, X_train, X_val,X_test
def renameProteins(cols_to_rename,somadict):
# Rename proteins

    new_cols = []
    for s in cols_to_rename:

        if 'seq' in s:
            if 'ratio' in s:
                s1 = s.split('_seq')[1]
                new_s1 = '-'.join(s1.split('.')[1:])
                try:
                    this_gene1 = somadict[somadict['SeqID'] == new_s1].loc[:, 'GeneID'].values[0]
                except:
                    this_gene2 = s1

                s2 = s.split('_seq')[2]
                new_s2 = '-'.join(s2.split('.')[1:])
                try:
                    this_gene2 = somadict[somadict['SeqID'] == new_s2].loc[:, 'GeneID'].values[0]
                except:
                    this_gene2 = s2
                new_cols.append(this_gene1 + '/' + this_gene2)
            else:
                try:
                    new_s = '-'.join(s.split('.')[1:])
                    this_gene = somadict[somadict['SeqID'] == new_s].loc[:, 'GeneID'].values[0]
                    new_cols.append(this_gene)
                except:
                    new_cols.append(s)
        else:
            # clinical feature
            new_cols.append(s)
    return new_cols
###########################################################################################################
###########################################################################################################
# INPUTS (fixed)

# Correlation threshold to include/exclude features
corr_thresh = 0.3

# Top features
useTopFeatures = False # Option to restrict the input to the following to features
top_features   = ['VWF/ADAMTS13', 'C5|C6/C7', 'Age', 'BMI']
topFeatsuffix  = '_topFeatures'


# Reduced Feature option
excludeAcuteCOVID19Features = False # Option to exclude severity associated clinical features


# Mutual proteins
useOnlyMutualProteins = False # Option for external validation comparison to reduce the number of proteins
data_file       = ''# load external validation data dictionary
feature_dict_df = pd.read_csv('your_feature_dict.csv')


# All other details
inputData       = '6M' # Options: '6M', '1M','1Mand6M'
useClinicalData = True # Option to include/exclude clinical features
n_splits        = 5
clf_choice      = 'RF' # Choice of classifier. Options: 'LR' = Logistic regression, 'RF' = Random Forest
endpoint        = 'PACS_12M'#6M_woDys' # Options: 'PACS_6M_woDys', 'PACS_12M'
usehealthy      = True # Options: True = include healty controls, False: no healty controls
doSHAP = False


# Paths
data_file_1M   = 'Data/Proteomics_Clinical_Data_220902_Acute_plus_healthy_v5.xlsx'
data_file_6M   = 'Data/Proteomics_Clinical_Data_220902_6M_timepoint_v4.xlsx'
label_file     = 'Data/Proteomics_Clinical_Data_220902_Labels_v2.xlsx'
output_folder  = 'Prediction_output'
partition_path = 'partitions'
createNewSplit = False


# Rename proteins to geneIDs
somadict = pd.read_excel('Data/SomaScanDict.xlsx')
somadict.columns = somadict.iloc[0, :]
somadict.drop(0, inplace=True)



###########################################################################################################
# Run prediction with feature selection

# Naming scheme
if usehealthy:
    healthy = 'withHealthy'
    partitionname = endpoint+'_partition_withHealthy'
else:
    healthy = 'noHealthy'
    partitionname = endpoint+'_partition_noHealthy'
if useClinicalData:
    feature_type = 'ClinicalProteomics'
else:
    feature_type = 'Proteomics'
name_save_all = endpoint+ '_from_'+inputData+feature_type+'_'+ healthy+ '_'+ clf_choice + '_withFCorr'
if useTopFeatures:
    name_save_all += '_'+topFeatsuffix
if useOnlyMutualProteins:
    name_save_all += '_' +"mutualProteins"
print('Working on '+ name_save_all )


# Output prep
output    = os.path.join(output_folder, name_save_all)
Path(output).mkdir(parents=True, exist_ok=True)
Path('partitions').mkdir(parents=True, exist_ok=True)

# Get label data
endpoint_label = pd.read_excel(label_file, index_col=0)
endpoint_label.set_index('SubjectID', inplace=True)
label = endpoint_label[endpoint]

# Get all features
data_1M = pd.read_excel(data_file_1M)
data_1M.set_index('SubjectID', inplace=True)
data_6M = pd.read_excel(data_file_6M)
data_6M.set_index('SubjectID', inplace=True)
data_healthy = data_1M[data_1M['COVID'] == 'Healthy']
data_1M = data_1M.drop(data_healthy.index)

# Get clinical data related to the COVID19 infection
cols_clinical6M = ['Age', 'Sex','Post_Vaccine']
cols_clinical_nonBin = ['Age', 'BMI','Acute_Nr_Symptoms']
cols_clinical = ['Age', 'Sex','Post_Vaccine','Asthma',
 'Lung','Diabetes','BMI','Cerebro','Heart',
 'Hypertonia','Autoimmune_diseases','Malignancy','Kidney','Fatigue',
 'Oxygen','Cough','Steroids','GI_symptoms','Remdesivir','Immuno',
 'ICU','Tocilizumab','Hydroxychloroquin','Dyspnoe','Allergic_disease',
 'Acute_Nr_Symptoms','Immunosuppressives','ACE_inhibitor','Fever']
cols_drop_fromFeatures = ['SampleId','Sampling_month','COVID',
                          'Days', 'Patient_Care','COVID19_Severity',
                          'COVID19_Severity_Grade','Index']
if excludeAcuteCOVID19Features:
    cols_clinical_keep = ['Age', 'Sex','Post_Vaccine','Asthma',
                         'Lung','Diabetes','BMI','Cerebro','Heart',
                         'Hypertonia','Autoimmune_diseases','Malignancy','Kidney','Allergic_disease']
    name_save_all += '_reduced'
else:
    cols_clinical_keep = cols_clinical

# Separate features, clinical data and lables
serverity_1M = data_1M[ ['Patient_Care','COVID19_Severity','COVID19_Severity_Grade']]
serverity_6M = data_6M[ ['Patient_Care','COVID19_Severity','COVID19_Severity_Grade']]
serverity_healthy = data_healthy[ ['Patient_Care','COVID19_Severity','COVID19_Severity_Grade']]


# Clinical data
data_clin_pats    = data_1M[cols_clinical]
data_clin_healthy = data_healthy[cols_clinical]
for c in cols_clinical:
    if c in cols_clinical_nonBin:
        data_clin_pats.loc[:, c] = data_1M.loc[:, c]
        data_clin_healthy.loc[:, c] = data_healthy.loc[:, c]
    else:
        data_clin_pats.loc[:,c] = data_1M.loc[:,c].map({'YES':1,'NO':0,'male':1,'female':0})
        data_clin_healthy.loc[:,c] = data_healthy.loc[:,c].map({'YES':1,'NO':0,'male':1,'female':0})
cols_not_not_in_healthy = list(set(data_clin_pats.columns)-set(data_clin_healthy.columns))
data_clin_healthy[cols_not_not_in_healthy] = 0


# Protein data
data_1M      = data_1M.drop(cols_clinical+cols_drop_fromFeatures, axis=1)
data_healthy = data_healthy.drop(cols_clinical+cols_drop_fromFeatures, axis=1)
data_6M      = data_6M.drop(cols_clinical6M + cols_drop_fromFeatures, axis=1)

# optional only use proteins appearing in reference data set
if useOnlyMutualProteins:

    seqIDs_keep   = []
    for s in feature_dict_df.SeqID:
        this_seqID = 'seq.'+('.').join(s.split('-'))
        seqIDs_keep.append(this_seqID)

    data_1M = data_1M.loc[:,seqIDs_keep]
    data_6M = data_6M.loc[:, seqIDs_keep]
    data_healthy = data_healthy.loc[:, seqIDs_keep]


# Get ratios of some proteins - note: Data are already log10 transformed!!!
if useOnlyMutualProteins:
    ratio1 = ['seq.2602.2', 'seq.3050.7', 'seq.4482.66']
    ratio2 = ['seq.2811.27', 'seq.3175.51', 'seq.2888.49']
else:
    ratio1 = ['seq.2602.2','seq.3050.7','seq.2381.52','seq.4482.66']
    ratio2 = ['seq.2811.27','seq.3175.51','seq.2888.49','seq.2888.49']
ratio_name = []
for i in range(0,len(ratio1)):
    ratio_name = 'ratio_'+ratio1[i]+ '_'+ratio2[i]
    data_1M[ratio_name] =  np.log10(10**(data_1M[ratio1[i]])/10**(data_1M[ratio2[i]]))
    data_6M[ratio_name] = np.log10(10**(data_6M[ratio1[i]]) / 10**(data_6M[ratio2[i]]))
    data_healthy[ratio_name] = np.log10(10**(data_healthy[ratio1[i]]) / 10**(data_healthy[ratio2[i]]))


# Now get the input data used in this run
serverity = serverity_1M
if inputData == '1M':
    data = data_1M
elif inputData == '6M':
    data = data_6M
elif inputData == '1Mand6M':
    cols_6M = [c+'_6M' for c in data_6M.columns]
    data_6M_app = data_6M.copy()
    data_6M_app.columns = cols_6M
    data_delta1M6M = data_1M-data_6M
    cols_1M6M= [c+'_1M-6M' for c in data_6M.columns]
    data_delta1M6M.columns = cols_1M6M

    # Concatenation of 1M and 6M data
    data    = pd.concat([data_1M,data_6M_app,data_delta1M6M], axis=1)
else:
    raise('Invalid choice of model inputData!')

# Include healthy controls if wanted
if usehealthy:
    if inputData == '1Mand6M':
        data_healthy_app = data_healthy.copy()
        data_healthy_app.columns = cols_6M
        data_healthy_delta1M6M = data_healthy-data_healthy
        data_healthy_delta1M6M.columns = cols_1M6M
        data_healthy = pd.concat([data_healthy, data_healthy_app, data_healthy_delta1M6M], axis=1)

    data = data.append(data_healthy)
    data_clin = data_clin_pats.append(data_clin_healthy)
    serverity = serverity.append(serverity_healthy)

# Check data and exclude patients with missing proteomics
data      = data.dropna()
data_clin = data_clin.loc[data.index,cols_clinical_keep]
label     = label.loc[data.index]
label     = label.dropna()
data      = data.loc[label.index]


# Exclude highly correlated features or useTopFeatures
if not useTopFeatures:

    # get correlation coefficients
    corr = data.corr()

    # Get absolute values for now, drop lower diagonal
    corr_abs = np.abs(np.triu(corr))

    # set diagonal to 0 to ignore
    np.fill_diagonal(corr_abs, 0)
    corr_abs_drop = corr_abs.copy()
    for c in tqdm(range(0,corr_abs.shape[0]-1)):
        if np.any(corr_abs_drop[c,:]>corr_thresh):
            corr_abs_drop[c, :] = 0
            corr_abs_drop[:, c] = 0
    feat_keep = corr.index[corr_abs_drop.sum(axis = 0)>0]
    data = data.loc[label.index, feat_keep]
    print('Retained '+str(len(feat_keep))+' protein features')

    # Rename features and save
    new_cols = renameProteins(data.columns, somadict)
    features_prot = pd.DataFrame(index=data.columns, columns = ['GeneID'])
    features_prot['GeneID'] = new_cols
    pd.DataFrame(features_prot).to_csv(os.path.join(output, name_save_all + '_ProtFeatures.csv'))
    data.columns = new_cols

else:

    # Rename features and save
    new_cols = renameProteins(data.columns, somadict)
    data.columns = new_cols


# Get splits or load
if createNewSplit:
    metadata = data_clin['Age'].to_frame()
    metadata['Patient_Care'] = serverity.Patient_Care.map({'Hospitalized': 1, 'Outpatient': 0, 'Mild':0, 'Healthy':0})
    metadata  = metadata.loc[label.index]
    data_clin = data_clin.loc[label.index]
    partition, labels = createSplit(label,metadata, n_splits = n_splits, stratifyFor = ['age_group','Patient_Care'])
    np.save(os.path.join(partition_path, partitionname + '_partition.npy'), partition)
    np.save(os.path.join(partition_path, partitionname + '_labelDic.npy'), labels)
else:
    partition = np.load(os.path.join(partition_path, partitionname + '_partition.npy'),
                        allow_pickle='TRUE').item()
    labels = np.load(os.path.join(partition_path, partitionname + '_labelDic.npy'),
                         allow_pickle='TRUE').item()


# Prepare outputs
if useTopFeatures:
    top_proteins = list(set(top_features) & set(data.columns))
    data = data.loc[:, top_proteins]

    top_clinical = list(set(top_features) & set(data_clin.columns))
    data_clin = data_clin.loc[:, top_clinical]

if useClinicalData:
    features = list(data.columns)+list(data_clin.columns)
else:
    features = list(data.columns)

pd.DataFrame(features).to_csv(os.path.join(output,name_save_all+'_features.csv'))

df_rank = pd.DataFrame(index=features, columns=np.arange(n_splits))
df_performance     = pd.DataFrame(index = np.arange(n_splits),columns = ['Prevalence','ROCAUC','AUPRC','relAPS'])



# Run prediction as nested CV
for cv in range(0, n_splits):
    print('Fold ' + str(cv))
    name_save = name_save_all + '_cv'+ str(cv)

    # Scale protein data
    if data.shape[1]>0:
        sc = StandardScaler()
        npx_reform_train = pd.DataFrame(sc.fit_transform(
            data.loc[partition['train' + str(cv)], :].values),
            index=data.loc[partition['train' + str(cv)], :].index,
            columns=data.columns)

        npx_reform_val = pd.DataFrame(sc.transform(
            data.loc[partition['validation' + str(cv)], :].values),
            index=data.loc[partition['validation' + str(cv)], :].index,
            columns=data.columns)

        npx_reform_test = pd.DataFrame(sc.transform(
            data.loc[partition['test' + str(cv)], :].values),
            index=data.loc[partition['test' + str(cv)], :].index,
            columns=data.columns)

        data_norm = npx_reform_train.append(npx_reform_val).append(npx_reform_test)
        data_norm = data_norm.loc[data.index, :]
    else:
        data_norm = pd.DataFrame(index=data.loc[partition['test' + str(cv)], :].index)

    # Clinical variables (if used)
    if useClinicalData:

        # Scale continous clinical features
        if 'Acute_Nr_Symptoms' in data_clin.columns:
            data_clin_cont = data_clin[['Age','BMI','Acute_Nr_Symptoms']]
        else:
            data_clin_cont = data_clin[['Age', 'BMI']]

        sc2 = StandardScaler()
        clin_cont_train = pd.DataFrame(sc2.fit_transform(
            data_clin_cont.loc[partition['train' + str(cv)], :].values),
            index=data.loc[partition['train' + str(cv)], :].index,
            columns=data_clin_cont.columns)

        clin_cont_val = pd.DataFrame(sc2.transform(
            data_clin_cont.loc[partition['validation' + str(cv)], :].values),
            index=data.loc[partition['validation' + str(cv)], :].index,
            columns=data_clin_cont.columns)

        clin_cont_test = pd.DataFrame(sc2.transform(
            data_clin_cont.loc[partition['test' + str(cv)], :].values),
            index=data.loc[partition['test' + str(cv)], :].index,
            columns=data_clin_cont.columns)

        # Combine again
        data_clin_cont = clin_cont_train.append(clin_cont_val).append(clin_cont_test)
        data_clin_norm = data_clin.copy()
        if 'Acute_Nr_Symptoms' in data_clin_norm.columns:
            data_clin_norm[['Age','BMI','Acute_Nr_Symptoms']] = data_clin_cont
        else:
            data_clin_norm[['Age', 'BMI']] = data_clin_cont

        # Impute BMI with 0 (since scales)
        data_clin_norm['BMI'] = data_clin_norm['BMI'].fillna(0)

        # Combine clinical and protein features
        data_norm = pd.concat([data_norm, data_clin_norm], axis=1)

    # Prediction
    if useTopFeatures:
        data_in = data_norm.loc[:,top_features].copy()
    else:
        data_in = data_norm.copy()
    best_est_cv, X_train, X_val,X_test = \
        getPredictionSingleFold_nooutput(cv, partition, labels,  clf_choice, data_in)

    # Save model and scalers
    try:
        joblib.dump(sc, os.path.join(output, name_save_all+'_cv'+str(cv)+'_scalerProteins.save'))
    except:
        print('no protein features')
    try:
        joblib.dump(sc2, os.path.join(output, name_save_all+'_cv'+str(cv)+'_scalerClinical.save'))
    except:
        print('no clinical features')
    pickle.dump(best_est_cv, open(os.path.join(output, name_save_all+'_cv'+str(cv)+'_model.save'), 'wb'))


    # Evaluate performance (all features)
    tn, fp, fn, tp, accuracy, precision, recall, \
    roc_auc, aps, f1 = perf_90recall(best_est_cv, X_test, [labels[p] for p in X_test.index])
    prev = np.mean([labels[p] for p in X_test.index])
    df_performance.loc[cv,:] = [prev,roc_auc,aps,aps/prev]


    # SHAP VALUES
    if doSHAP:
        if clf_choice == 'LR':
            explainer = shap.Explainer(best_est_cv, X_train, feature_names=X_train.columns)
        elif clf_choice == 'MLP' or clf_choice == 'lightGBM':
            explainer = shap.Explainer(best_est_cv.predict, X_train, feature_names=X_train.columns)
        elif clf_choice == 'RF':
            explainer = shap.TreeExplainer(best_est_cv, X_train, feature_names=X_train.columns,check_additivity=False)
        if clf_choice == 'RF':
            shap_values = explainer(X_train, check_additivity=False)
        else:
            shap_values = explainer(X_train)

        # Now pick out the highest shap values
        feature_names = shap_values.feature_names
        try:
            shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        except:
            shap_df = pd.DataFrame(shap_values.values[:, :, 0], columns=feature_names)
        vals = np.abs(shap_df.values).mean(0)
        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                       columns=['col_name', 'feature_importance_vals'])
        shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        shap_importance.to_csv(os.path.join(output, name_save_all+'_cv'+str(cv) + '_importance_val_all.csv'))

        # SHAP Plot for this fold
        plt.figure()
        try:
            shap.summary_plot(shap_values, X_train, show=False)
        except:
            shap.summary_plot(shap_values[:, :, 0], X_train, show=False)
        fig =  plt.gcf()
        plt.tight_layout()
        fig.savefig(os.path.join(output, name_save_all+'_cv'+str(cv) + '_SHAP_all.pdf'),format = 'pdf')


        # Get feture importance ranking
        this_rank = shap_importance.set_index('col_name').rank(ascending=False)
        df_rank[cv] = this_rank.loc[df_rank.index]#df_rank.sort_values(ascending=False, by=cv)


# Save results to df
df_performance.to_csv(os.path.join(output,name_save_all +'_performance.csv'))











