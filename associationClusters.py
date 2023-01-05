' Script to for association analysis of clusters. Created by S Brueningk 2023 '

import pandas as pd
import numpy as np
from IPython.core.display import display, Markdown
import os.path
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st

###########################################################################################################
# Functions
def logistic_regression(X, X_additional, Y, n_proteins):
    pvals = []
    for i in range(n_proteins):  # a protein each time
        X_i = X[:, i]
        X_tot = np.c_[X_additional, X_i]
        model = sm.Logit(Y, X_tot).fit(disp=0)
        pvals.append(model.pvalues[-1])
    return pvals
def qqplot(pvals, phenotype, endpoint, alpha=0.05, figsize=[7, 5]):
    plt.figure(figsize=figsize)

    maxval = 0
    M = pvals.shape[0]
    pnull = np.arange(1, M + 1) / M  # uniform distribution for the pvals
    # Taking the log10 of expected and observed
    qnull = -np.log10(pnull)
    qemp = -np.log10(np.sort(pvals))

    # Taking medians and plotting it
    qnull_median = np.median(qnull)
    qemp_median = np.median(qemp)

    xl = r'$-log_{10}(P)$ observed'
    yl = r'$-log_{10}(P)$ expected'
    if qnull.max() > maxval:
        maxval = qnull.max()
    plt.plot(qnull, qemp, 'o', markersize=2)
    plt.plot([0, qnull.max()], [0, qnull.max()], 'k')
    plt.ylabel(xl)
    plt.xlabel(yl)

    betaUp, betaDown, theoreticalPvals = qqplot_bar(M=M, alphalevel=alpha)
    lower = -np.log10(theoreticalPvals - betaDown)
    upper = -np.log10(theoreticalPvals + betaUp)
    plt.fill_between(-np.log10(theoreticalPvals), lower, upper, color="grey", alpha=0.5)

    plt.axis(xmin=-0.15, ymin=-0.15)
    plt.title(endpoint, fontsize=20)
    plt.tight_layout()
    plt.show()
    return 0
def qqplot_bar(M=1000000, alphalevel=0.05, distr='log10'):
    mRange = 10 ** (np.arange(np.log10(0.5), np.log10(M - 0.5) + 0.1, 0.1));  # should be exp or 10**?
    numPts = len(mRange);
    betaalphaLevel = np.zeros(numPts);  # down in the plot
    betaOneMinusalphaLevel = np.zeros(numPts);  # up in the plot
    betaInvHalf = np.zeros(numPts);
    for n in range(numPts):
        m = mRange[n];  # numplessThanThresh=m;
        betaInvHalf[n] = st.beta.ppf(0.5, m, M - m);
        betaalphaLevel[n] = st.beta.ppf(alphalevel, m, M - m);
        betaOneMinusalphaLevel[n] = st.beta.ppf(1 - alphalevel, m, M - m);
        pass
    betaDown = betaInvHalf - betaalphaLevel;
    betaUp = betaOneMinusalphaLevel - betaInvHalf;

    theoreticalPvals = mRange / M;
    return betaUp, betaDown, theoreticalPvals
def manhattan_plot(pvals, alpha, phenotype, title):
    logpval = -np.log10(np.array(pvals).astype(float))
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(len(pvals)), logpval, ls='', marker='.', color='k')
    plt.tick_params(axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

    plt.hlines(-np.log10(alpha / (2 * len(pvals))), -0.1, len(pvals) + 0.1, color='red')
    plt.xlabel('proteins')
    plt.ylabel('-log10(p-values)')
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()
def significant_proteins(pvals, proteins, alpha):
    index = pvals <= (alpha / (2 * len(proteins)))
    associated_proteins = proteins[index]
    print('\nASSOCIATED PROTEINS\n----------------\nProtein\t\tp-value')
    for ass, pv in zip(associated_proteins, pvals[index]):
        print('{}\t{}'.format(ass, pv))
def significant_proteins_5per(pvals, alpha):
    N = int(len(pvals))
    pvals = pvals.sort_values().iloc[:N]
    print('lowest p-values:{:.3g},\tthreshold:{:.3g} '.format(pvals.min(), alpha / (2 * N)))
    corrected_pvals = pvals[pvals <= (alpha / (2 * N))]
    associated_proteins = corrected_pvals.index.tolist()
    # print('\nASSOCIATED PROTEINS\n----------------\nProtein\t\tp-value')
    # for ass, pv in zip(associated_proteins, corrected_pvals.values):
    # print('{}\t{}'.format(ass, pv))
    return associated_proteins
def linear_regression(X, X_additional, Y, n_proteins):
    pvals = []
    for i in range(n_proteins):  # a protein each time
        X_i = X[:, i]
        X_tot = np.c_[X_additional, X_i]
        model = sm.OLS(Y, X_tot).fit()
        pvals.append(model.pvalues[-1])
    return pvals
def qq_plot_all(all_pvals, all_title):
    fig, axes = plt.subplots(nrows=1, ncols=len(all_title),
                             figsize=(5, 5), sharex=True, sharey=True)
    M = all_pvals[0].shape[0]
    pnull = np.arange(1, M + 1) / M
    qnull = -np.log10(pnull)
    # qemp = -np.log10(np.sort(pvals))
    # alpha = 0.5
    xl = r'$-log_{10}(P)$ observed'
    yl = r'$-log_{10}(P)$ expected'

    try:
        for i, ax in enumerate(axes.flatten()):
            if i >= len(all_pvals):
                continue
            qemp = -np.log10(np.sort(all_pvals[i]))
            ax.plot(qnull, qemp, 'o', markersize=2)
            ax.plot([0, qnull.max()], [0, qnull.max()], 'k')
            ax.axis(xmin=-0.15, ymin=-0.15)
            ax.set_title(all_title[i], fontsize=20)
    except:
        i = 0
        qemp = -np.log10(np.sort(all_pvals[i]))
        axes.plot(qnull, qemp, 'o', markersize=2)
        axes.plot([0, qnull.max()], [0, qnull.max()], 'k')
        axes.axis(xmin=-0.15, ymin=-0.15)
        axes.set_title(all_title[i], fontsize=20)


    fig.supxlabel(xl, fontsize=20, y=0)
    fig.supylabel(yl, fontsize=20, x=0)
    #fig.suptitle('Processing: ' + processing, fontsize=30, y=1.05)
    fig.tight_layout()
    fig.show()
def manhattan_plot_all(all_pvals, alpha, all_titles, perc_thres):
    fig, axes = plt.subplots(nrows=1, ncols=len(all_titles),
                             figsize=(20, 9), sharex=True, sharey=True)

    try:
        for i, ax in enumerate(axes.flatten()):
            if i >= len(all_pvals):
                continue
            N = int(len(all_pvals[i]) * perc_thres)
            pvals_selected = all_pvals[i].sort_values().iloc[:N]
            pvals_selected = pvals_selected.sort_index()
            logpval = -np.log10(
                np.array(pvals_selected).astype(float))
            ax.plot(np.arange(len(pvals_selected)),
                    logpval, ls='', marker='.', color='k')
            ax.hlines(
                -np.log10(alpha / (2 * len(pvals_selected))),
                -0.1, len(pvals_selected) + 0.1, color='red')
            ax.set_title(all_titles[i], fontsize=20)
    except:
        i = 0
        N = int(len(all_pvals[i]) * perc_thres)
        pvals_selected = all_pvals[i].sort_values().iloc[:N]
        pvals_selected = pvals_selected.sort_index()
        logpval = -np.log10(
            np.array(pvals_selected).astype(float))
        axes.plot(np.arange(len(pvals_selected)),
                logpval, ls='', marker='.', color='k')
        axes.hlines(
            -np.log10(alpha / (2 * len(pvals_selected))),
            -0.1, len(pvals_selected) + 0.1, color='red')
        axes.set_title(all_titles[i], fontsize=20)

    fig.supxlabel('Proteins', fontsize=20, y=0)
    fig.supylabel(r'$-log_{10}(P)$', fontsize=20, x=0)
    #fig.suptitle('Processing: {}, top {:.0f}%'.format(processing, perc_thres * 100), fontsize=30, y=1.05)
    fig.tight_layout()
    fig.show()
###########################################################################################################
##########################################################################################################
# INPUTS (fixed)
useScaler       = True
stratfield      = 'Patient_Care'  # Account for COVID19 severity
stratify_for    = ['age_group', stratfield]
inputData       = '6M'            # Options: '6M', '1M','1Mand6M'
endpoint        = 'PACS_6M_woDys' # Options: 'PACS_6M_woDys', 'PACS_12M'
usehealthy      = True            # Options: True = include healthy controls, False: no healthy controls

# Use reduced feature set in case of external validation
reduceFeaturestoexternal = True
external_features_keep    = 'Data/external_usedClustersAndProteins.csv'

# Paths
data_file_1M     = 'Data/Proteomics_Clinical_Data_220902_Acute_plus_healthy_v5.xlsx'
data_file_6M     = 'Data/Proteomics_Clinical_Data_220902_6M_timepoint_v4.xlsx'
label_file       = 'Data/Proteomics_Clinical_Data_220902_Labels_v2.xlsx'
output_folder    = 'Association_output'
protein_clusters = 'Data/Table S2 Biological protein cluster compositions.xlsx'

do_clusterAssociation = True
do_singleProteinAssociation = False

###########################################################################################################
# Run
# Prepare output
df_threshs = pd.DataFrame(columns = ['sig thresh'])
output     = os.path.join(output_folder)
Path(output).mkdir(parents=True, exist_ok=True)


# Create name
if usehealthy:
    healthy = 'withHealthy'
    name = endpoint+'_withHealthy'
else:
    healthy = 'noHealthy'
    name = endpoint+'_noHealthy'
print('Working on '+ name)


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
cols_clinical_keep = cols_clinical


# Separate features used for association analysis
severity_1M = data_1M[ ['Patient_Care','COVID19_Severity','COVID19_Severity_Grade']]
severity_6M = data_6M[ ['Patient_Care','COVID19_Severity','COVID19_Severity_Grade']]
severity_healthy = data_healthy[ ['Patient_Care','COVID19_Severity','COVID19_Severity_Grade']]


# Clinical data (here only age and sex are used)
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


# Get ratios of some proteins - note: Data are already log10 transformed!!!
ratio1 = ['seq.2602.2','seq.3050.7','seq.2381.52','seq.4482.66']
ratio2 = ['seq.2811.27','seq.3175.51','seq.2888.49','seq.2888.49']
ratio_name = []
for i in range(0,len(ratio1)):
    ratio_name = 'ratio_'+ratio1[i]+ '_'+ratio2[i]
    data_1M[ratio_name] =  np.log10(10**(data_1M[ratio1[i]])/10**(data_1M[ratio2[i]]))
    data_6M[ratio_name] = np.log10(10**(data_6M[ratio1[i]]) / 10**(data_6M[ratio2[i]]))
    data_healthy[ratio_name] = np.log10(10**(data_healthy[ratio1[i]]) / 10**(data_healthy[ratio2[i]]))


# Now get the input data used in this run
severity = severity_1M
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
    severity = severity.append(severity_healthy)

# Check data and exclude patients with missing proteomics
data      = data.dropna() # Should not make any differene for our data
data_clin = data_clin.loc[data.index,cols_clinical_keep]
label     = label.loc[data.index]
label     = label.dropna()
data      = data.loc[label.index]



# Scale each protein (as used later)
sc = StandardScaler()
npx_reform_train = pd.DataFrame(sc.fit_transform(
    data.loc[:, :].values),
    index=data.loc[:, :].index,
    columns=data.columns)


# scale age
COVs = data_clin[['Age', 'Sex']]
COVs[stratfield] = severity.loc[data_clin.index, stratfield].map(dict(Outpatient=0,
                                                                     Hospitalized=1,
                                                                     Healthy=0))
scaler = StandardScaler()
COVs_sc = COVs.copy()
COVs_sc['Age'] = scaler.fit_transform(COVs_sc['Age'].values.reshape(-1, 1))


# Prepare
n, n_proteins = npx_reform_train.shape
X_additional  = np.c_[np.ones(n), COVs_sc.values]
X             = npx_reform_train.values
phenotype     = label.values.astype(np.float64)

if do_singleProteinAssociation:
    # Single protein association
    if len(np.unique(phenotype)) == 2:
        pvals = logistic_regression(X, X_additional, phenotype, n_proteins)
    else:
        pvals = linear_regression(X, X_additional, phenotype, n_proteins)
    pvals = pd.Series(pvals, index=npx_reform_train.columns)
else:
    pvals = pd.Series()


# Clusters
if do_clusterAssociation:
    df_prots = pd.read_excel(protein_clusters)

    if reduceFeaturestoexternal:
        df_prots_external = pd.read_csv(external_features_keep, index_col=0)

        missing = []
        features_keep = []
        keep_index = []
        for p in df_prots_external.index:
            if p not in df_prots['AptamerName'].values:
                missing.append(p)
            else:
                features_keep.append(p)
                keep_index += list(df_prots[df_prots['AptamerName'] == p].index)

        df_prots = df_prots.loc[np.unique(keep_index)]



    clusters     = df_prots['Group'].unique()
    for this_cl in clusters:

        # get protein group
        ids_use = list(df_prots[df_prots['Group']==this_cl]['AptamerName'].values)
        prots_use = [p for p in ids_use]


        # get the relevant data
        X_i = npx_reform_train.loc[:, prots_use].values
        X_tot = np.c_[X_additional, X_i]

        # Association
        try:
            model = sm.Logit(phenotype, X_tot).fit(disp=0, method='bfgs')
            pvals.loc[this_cl] = model.llr_pvalue
            print(this_cl + ': ' + str(model.llr_pvalue))
        except:
            print('Failed Single Group ' + this_cl)
            pvals.loc[this_cl] = np.nan

# Organize and save results
alpha = 0.05
sig_asso = significant_proteins_5per(pvals, alpha)
display(Markdown('Associated protein: **{}**'.format(sig_asso)))
pvals = pvals.sort_values()
pvals.sort_values().to_csv(os.path.join(output_folder, name  + '_singleSomamer_pvals.csv'))


# Plot and save
perc_thres = 0.05
this_sig_thresh = alpha/(2*len(pvals))
qq_plot_all([pvals], [name])
plt.savefig(os.path.join(output_folder, name  + '_qq_plot.png'))
manhattan_plot_all([pvals], alpha, [name], perc_thres)
plt.savefig(os.path.join(output_folder, name + '_manhattan_plot5perc.png'))
df_threshs.loc[name,'sig thresh'] =  this_sig_thresh
df_threshs.to_csv(os.path.join(output_folder, name  +'_thresholds.csv'))







