' Script to for univariate association analysis of proteomics with LongCOVID label. Created by S Brueningk 2023 '
import pandas as pd
import numpy as np
from IPython.core.display import display, Markdown
import os.path
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")


###########################################################################################################
def create_split(clinical, binendpoints, n_splits, stratify_for, this_endpoint, allData, printPrev=True,
                 sevfield='CareSamp'):  # 'SevSamp2'):
    if 'age_group' in stratify_for:
        age_groups_max = [20, 50, 70, 150]
        age_group = []
        for s in clinical.index:
            this_age = clinical.loc[s, 'Age']
            age_group.append(str(age_groups_max[np.where(np.array(age_groups_max) >= this_age)[0][0]]))

    if sevfield in stratify_for:
        sevmax = allData.loc[clinical.index, sevfield]

    concatLabels = [str(i) for i in binendpoints.loc[clinical.index, this_endpoint]]
    for i, var in enumerate(stratify_for):
        if var == sevfield:
            vals = sevmax
        elif var == 'age_group':
            vals = age_group
        else:
            vals = clinical.loc[:, var].values
        concatLabels = [concatLabels[i] + str(vals[i]) for i in range(0, len(vals))]

    from sklearn.model_selection import StratifiedKFold
    from collections import Counter
    skf = StratifiedKFold(n_splits=n_splits)
    split = 0
    partition = dict()
    for train_index, test_index in skf.split(pd.Series(clinical.index), concatLabels):
        partition['train' + str(split)] = pd.Series(clinical.index)[train_index].values
        partition['test' + str(split)] = pd.Series(clinical.index)[test_index].values

        if printPrev:
            print('Prevalence train/test split ' + str(split) + ':')
            print(binendpoints.loc[partition['train' + str(split)], this_endpoint].sum() / len(
                partition['train' + str(split)]))
            print(binendpoints.loc[partition['test' + str(split)], this_endpoint].sum() / len(
                partition['test' + str(split)]))

        split = split + 1

    labels_dic = dict(zip(list(clinical.index), binendpoints.loc[clinical.index, this_endpoint]))
    return partition, labels_dic
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
# INPUTS (fixed)
useScaler        = True
stratfield       = 'CareSamp'  # 'SevSamp2'
stratify_for     = ['age_group', stratfield]

names = [
'PACS_1M_whealthy',
'PACS_1M_woDys_whealthy',
'PACS_6M_whealthy_use1Monly',
'PACS_6M_woDys_whealthy_use1Monly'
]

n_splits         = 5
useScaler        = True


# paths
path_allData  = '/cluster/work/borgw/Long_COVID/Data_Boyman_Borgwardt'
path          = '/cluster/work/borgw/Long_COVID/Data_Boyman_Borgwardt/Preprocessed'
output_path   = '/cluster/work/borgw/Long_COVID/output/output_associationALL'


###########################################################################################################

# Run
allData = pd.read_csv(os.path.join(path_allData, 'Proteomics_Clinical_Data_Clean_211213_CC10.csv'))
allData.set_index('SampleId', inplace=True, drop=True)


df_threshs = pd.DataFrame(columns = ['sig thresh'])
for name in names:

    print('Working on '+ name)

    # Get data
    endpoint_label = pd.read_csv(os.path.join(path, name + '_labels.csv'), index_col=0)
    clinical_in = pd.read_csv(os.path.join(path, name + '_clinical.csv'), index_col=0)
    this_proteins = pd.read_csv(os.path.join(path, name + '_protein.csv'), index_col=0)

    if 'SubjectID' in this_proteins.columns:
        this_proteins.drop('SubjectID', axis=1, inplace=True)

    partition = np.load(os.path.join(path, name + '_partition.npy'), allow_pickle='TRUE').item()
    labels = np.load(os.path.join(path, name + '_labelDic.npy'), allow_pickle='TRUE').item()

    prevs = []
    for i in range(n_splits):
        lables_thisSplit = [labels[partition['test' + str(i)][j]] for j in range(len(partition['test' + str(i)]))]
        prevs.append(np.sum(lables_thisSplit) / len(lables_thisSplit))

    meanPrv = np.mean(prevs)
    stdPrv = np.std(prevs)


    COVs_all = clinical_in[['Age', 'Sex']]
    if stratfield in ['SevMax2', 'SevSamp2']:
        COVs_all[stratfield] = allData.loc[clinical_in.index, stratfield].map(
            dict(Mild=0, Severe=1, Healthy=0))
    else:
        COVs_all[stratfield] = allData.loc[clinical_in.index, stratfield].map(dict(Outpatient=0,
                                                                                   Hospitalized=1,
                                                                                   Healthy=0))

    # Scale each protein (as used later)
    sc = StandardScaler()
    npx_reform_train = pd.DataFrame(sc.fit_transform(
        np.log10(this_proteins.loc[:, :].values)),
        index=this_proteins.loc[:, :].index,
        columns=this_proteins.columns)


    # get protein data for each fold
    COVs = COVs_all

    # scale age
    scaler = StandardScaler()
    COVs_sc = COVs.copy()
    COVs_sc['Age'] = scaler.fit_transform(COVs_sc['Age'].values.reshape(-1, 1))

    n, n_proteins = npx_reform_train.shape
    print(n_proteins)
    X_additional = np.c_[np.ones(n), COVs_sc.values]
    X = npx_reform_train.values
    phenotype = endpoint_label.values.astype(np.float64)

    if len(np.unique(phenotype)) == 2:
        pvals = logistic_regression(X, X_additional, phenotype, n_proteins)
    else:
        pvals = linear_regression(X, X_additional, phenotype, n_proteins)
    pvals = pd.Series(pvals, index=npx_reform_train.columns)
    sig_asso = significant_proteins_5per(pvals, 0.05)
    display(Markdown('Associated protein: **{}**'.format(sig_asso)))

    Path(output_path).mkdir(parents=True, exist_ok=True)
    pvals.sort_values().to_csv(os.path.join(output_path, name  + '_pvals.csv'))

    # Plot
    alpha = 0.05
    this_sig_thresh = alpha/(2*len(pvals))
    qq_plot_all([pvals], [name])
    plt.savefig(os.path.join(output_path, name  + '_qq_plot.png'))
    perc_thres = 0.05
    manhattan_plot_all([pvals], alpha, [name], perc_thres)
    plt.savefig(os.path.join(output_path, name + '_manhattan_plot5perc.png'))
    perc_thres = 1
    manhattan_plot_all([pvals], alpha, [name], perc_thres)
    plt.savefig(os.path.join(output_path, name + '_manhattan_plotAll.png'))

    df_threshs.loc[name,'sig thresh'] =  this_sig_thresh
    df_threshs.to_csv(os.path.join(output_path, 'threshopds.csv'))








