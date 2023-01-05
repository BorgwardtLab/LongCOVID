import os
import pandas as pd

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


# Which data should be evaluated
output   = 'Prediction_output'

model = 'RF'
names    = ['PACS_6M_woDys_from_6MProteomics_withHealthy_RF_withFCorr_mutualProteins']
# 'PACS_6M_woDys_from_1MClinicalProteomics_withHealthy_'+model+'_withFCorr',
# 'PACS_6M_woDys_from_6MClinicalProteomics_withHealthy_'+model+'_withFCorr',
# 'PACS_6M_woDys_from_1Mand6MClinicalProteomics_withHealthy_'+model+'_withFCorr',
# 'PACS_12M_from_6MClinicalProteomics_withHealthy_'+model+'_withFCorr']

n_splits = 5
somadict = pd.read_excel('Data/SomaScanDict.xlsx')
somadict.columns = somadict.iloc[0,:]
somadict.drop(0,inplace = True)

for name in names:
    folder = os.path.join(output,name)

    # Load shap analysis results
    this_importance_all = pd.read_csv(os.path.join(folder, name +'_cv'+str(1) + '_importance_val_all.csv'), index_col=0)
    this_importance_all.set_index('col_name', inplace=True)
    df_importance = pd.DataFrame(index = this_importance_all.index, columns = [cv for cv in range(0,n_splits)])
    for cv in range(0,n_splits):
        this_importance = pd.read_csv(os.path.join(folder, name+'_cv'+str(cv) + '_importance_val_all.csv'), index_col=0)
        this_importance.set_index('col_name', inplace=True)
        df_importance.loc[this_importance.index,cv] = this_importance.loc[:,'feature_importance_vals'].rank(ascending = False)

    df_importance['sum'] = df_importance.loc[:,[0,1,2,3,4]].sum(axis =1)
    df_importance= df_importance.sort_values('sum',ascending = True)

    # Column names
    new_cols = renameProteins(df_importance.index,somadict)
    df_importance.index = new_cols
    df_importance.to_csv(os.path.join(output,'eval', name +'_importance_val_all.csv'))