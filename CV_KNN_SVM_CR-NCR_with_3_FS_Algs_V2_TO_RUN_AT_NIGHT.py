''' K-FOLD EXERCISE - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_ind
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLasso

''' Important Constants '''
N_PVAL_GENES = 50
SIG_LEVEL = 0.001
N_ITERATIONS = 100
K_FOLDS = 5
N_NEIGHBORS = 17

''' Gene selection function (most DE genes) '''
def n_genes_p_value(n_genes, X, y, alpha):
    gene_to_pval = {}
    
    print('getting',n_genes,'genes with lowest p-value')
    
    CR_samples = X.loc[[s for s in y.index if y.loc[s][0] == 'CR']]
    not_CR_samples = X.loc[[s for s in y.index if y.loc[s][0] == 'Not in CR']]
    
    for gene in X.columns:
        p_value = ttest_ind(CR_samples[gene], not_CR_samples[gene])[1]
        if (p_value <= alpha):
            gene_to_pval[gene] = p_value
  
    return sorted(gene_to_pval, key=lambda x: gene_to_pval[x])[:n_genes]

''' Function for CV with Different Classifiers for a given classifier '''
def genes_aucs_FS(X, y, folds, clf, fs_func, N_PVAL_GENES):
    fold = 1
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    gene_to_auc = {}
    
    for train_index, test_index in skf.split(X, y):
        
        print('Starting fold', fold, 'in genes_aucs_FS')
        
        ''' Reseting scores & labels for this fold's AUC computation '''    
        scores, labels = [], []
        FS_genes = []
        
        ''' Fetch train and validation subsets '''
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
        
        ''' Keep only n_genes with lowest p-value '''
        p_value_genes = n_genes_p_value(N_PVAL_GENES, X_train, y_train, SIG_LEVEL)
        
        rfe = '' # you need this cause you might use this to fit and predict
        probas = []
        
        if (fs_func == 'HC'):
            ''' Split this fold's train subset (4/5) to -> (4/5) and (1/5)'''
            X_sub_train, X_sub_test, y_sub_train, y_sub_test = \
                train_test_split(X_train[p_value_genes], y_train, test_size=0.20,
                random_state=42, stratify=y_train)
            
            ''' Get best genes by Hill Climbing on (4/5) out of fold's 4/5 '''
            FS_genes = hill_climbing(X_sub_train, X_sub_test, y_sub_train, y_sub_test, clf)
            
        elif (fs_func == 'RFE'):
            print('RFE')
            rfe = RFE(clf['model'], N_PVAL_GENES)
            rfe.fit(X_train[p_value_genes],
                    [1 if y == 'CR' else 0 for y in y_train.values])
            FS_genes = [p_value_genes[i] for i in range(len(rfe.ranking_)) 
                        if rfe.ranking_[i] == 1]
        else:
            print('LASSO')
            rlasso = RandomizedLasso(selection_threshold=0.1)
            rlasso.fit(X_train[p_value_genes],
                       [1 if y == 'CR' else 0 for y in y_train.values])
            FS_genes = [p_value_genes[i] for i in rlasso.get_support(True)]
            if (len(FS_genes) == 0):
                d = dict(zip(p_value_genes, rlasso.scores_))
                FS_genes = sorted(d, key=lambda x: d[x])[:5]
        
        ''' fit and predict (---using genes picked---)'''
        if (fs_func == 'RFE'):
            probas = rfe.predict_proba(X_test[p_value_genes])
        else:
            clf['model'].fit(X_train[FS_genes],
                             [1 if y == 'CR' else 0 for y in y_train.values])
            probas = clf['model'].predict_proba(X_test[FS_genes])
        
        scores = [p[clf['pos_idx']] for p in probas]
        labels = [1 if y == 'CR' else 0 for y in y_test.values]
        
        ''' Computing this fold's AUC '''
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        print('Clf: {}, Fold {}, AUC={}'.format(clf['name'], fold, auc))
        
        ''' Adding Gene AUC Scores '''
        for gene in FS_genes:
            if gene in gene_to_auc:
                gene_to_auc[gene].append(auc)
            else:
                gene_to_auc[gene] = [auc]
        
        fold += 1 # Incrementing fold count
        
    return gene_to_auc

def cv_kfold(X, y, folds, clf):
    scores, labels = [], []
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        ''' Get train and test data '''
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ''' fit and predict '''
        clf['model'].fit(X_train,
                         [1 if y == 'CR' else 0 for y in y_train.values])
        probas = clf['model'].predict_proba(X_test)
        scores += [p[clf['pos_idx']] for p in probas]
        labels += [1 if y == 'CR' else 0 for y in y_test.values]
        
    return scores, labels

def hill_climbing(X_train, X_test, y_train, y_test, clf):
    
    print('hill climbing')
    
    percent = 0.001
    genes_picked = []
    genes_not_picked = X_train.columns.tolist()
    improvement = 10
    auc_climb = 0.0001
    
    ''' While we can still improve the auc_climb in XXX% (let's say 10%) '''
    while improvement >= (1 + percent):
        gene = ''
        max_auc = 0
        
        ''' find gene with highest auc when added to genes_picked '''
        for g in genes_not_picked:
            ''' Calculate auc '''
            clf['model'].fit(X_train[genes_picked + [g]],
                             [1 if y == 'CR' else 0 for y in y_train.values])
            probas = clf['model'].predict_proba(X_test[genes_picked + [g]])
            scores = [p[clf['pos_idx']] for p in probas]
            labels = [1 if y == 'CR' else 0 for y in y_test.values]
            fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            
            ''' if resulting auc is bigger than current max, change max '''
            if auc > max_auc:
                max_auc = auc
                gene = g
        
        improvement = max_auc / auc_climb
        
        ''' if improvement is >= percent%, add this gene '''
        if improvement >= (1 + percent):
            auc_climb = max_auc
            genes_picked.append(gene)
            genes_not_picked.remove(gene)
            
    print(len(genes_picked),'HC genes: \n', 
          gid_to_name[genes_picked].values.flatten().tolist())
    return genes_picked

''' ------------------------- MAIN PROGRAM -------------------------------- '''
''' Importing Clinical Info '''
print('importing AML_CLINICAL')
clinical_df = pd.read_csv('C:/Users/user/Desktop/AML_CLINICAL.csv', index_col=[0])

''' Importing data and creating data frame containing only AML samples '''
print('importing AML_NBM')
df = pd.read_csv('C:/Users/user/Desktop/NBM_AML.txt',
                 delimiter='\t', header=[0], index_col=[1]).T.iloc[21:,:]

''' Shortening AML data's index names '''
df = df.rename(lambda idx: idx[:6], axis='index')

''' dropping duplicates in AML data (not keeping any)'''
df = df.reset_index(0).drop_duplicates('index', keep=False).set_index('index')

''' Finding the intersection '''
print('Finding intersection between AML data and clinical data') 
intersection = set(df.index).intersection(set(clinical_df.index))

''' Removing undocumented samples and sorting data frames by name '''
df = df.loc[intersection].sort_index()
clinical_df = clinical_df.loc[intersection].sort_index()

''' --------------- END OF FETCHING AND ORGANIZING DATA ------------------- '''

''' Fetching gene id to gene name mapping '''
gid_to_name = pd.read_csv('C:/Users/user/Desktop/NBM_AML.txt', delimiter='\t',
                          index_col=[1]).T.iloc[:1]

''' Extracting BALANCED target set from clinical_df (at 2nd CR category) '''

ncr_df = clinical_df.loc[clinical_df['CR status at end of course 2'] == 'Not in CR']
cr_df = clinical_df.loc[clinical_df['CR status at end of course 2'] == 'CR']
cr_ncr_df = cr_df.iloc[:len(ncr_df)].append(ncr_df)[['CR status at end of course 2']]
cr_ncr_df = cr_ncr_df.sort_index()

''' Extracting BALANCED data set from df '''
X_df = df.loc[cr_ncr_df.index]

''' Normalizing data for SVM '''
from sklearn.preprocessing import normalize
X_df = pd.DataFrame(data=normalize(X_df, norm='max'), columns=X_df.columns,
                    index=X_df.index)

''' IMPORTANT - CAREFUL - KEEEP random_state equal to 42 (meaning of life) '''
''' Splitting into train and test subsets (setting test set aside till Poster) '''
X_big_train, X_final_test, y_big_train, y_final_test = train_test_split(X_df,
                                                                cr_ncr_df,
                                                                test_size=0.20,
                                                                random_state=42,
                                                                stratify=cr_ncr_df)

''' Instantiating classifier '''
svc = svm.SVC(kernel='linear', probability=True)
knn = KNeighborsClassifier(N_NEIGHBORS)

''' Instantiating list of classifiers'''
classifiers = [
              {
               'name': 'KNN',
               'model': knn,
               'pos_idx': 1
              },{
               'name': 'SVM',
               'model': svc,
               'pos_idx': 0
              }
              ]

''' keeping log of genes'''
log_best_genes = ''

''' keeping scores of genes '''
gta_scores = []

''' CV with 3 FS algorithms ROC graph for each FS (with KNN & SVM) '''
for fs_func in ['HC', 'RFE','R.LASSO']:

    print('Starting with FS function =', fs_func)
    
    plt.figure(figsize=(8, 6))
    
    for clf in classifiers:
        
        if (clf['name'] == 'KNN' and fs_func == 'RFE'):
            continue
        
        print('Starting', clf['name'],'Classifier')
        
        genes_to_AUCs = {}
                    
        for i in range(N_ITERATIONS):
            
            print('Iteration',i+1,'\n\tCalling genes_aucs_FS()')
            
            ''' Get gene_to_auc dict from genes_aucs_FS() '''
            FS_dict = genes_aucs_FS(copy.deepcopy(X_big_train),
                                    copy.deepcopy(y_big_train), 
                                    K_FOLDS, clf, fs_func, N_PVAL_GENES)
            
            ''' Adding Gene AUC Scores '''
            for gene in FS_dict:
                if gene in genes_to_AUCs:
                    genes_to_AUCs[gene] += FS_dict[gene]
                else:
                    genes_to_AUCs[gene] = FS_dict[gene]
        
        gta_scores.append(genes_to_AUCs)
        
        ''' Sort best performing genes '''
        sorted_best_genes = sorted(genes_to_AUCs, reverse=True,
                           key=lambda x: np.mean(genes_to_AUCs[x]))
        
        for n_final_genes in [5, 10, 15, 20, 30]:
                
            print('Computing final "training AUC" (5-fold cv) for N_ITERATIONS')
            log_best_genes += '\n Best ' + str(n_final_genes) + ' genes for ' \
                              + clf['name']+' with '+fs_func+' are:\n' \
                              + str(gid_to_name[sorted_best_genes[:n_final_genes]].values.flatten().tolist())
            print(log_best_genes)
    
            print('Starting K-FOLD CV with',n_final_genes,'best genes')
        
            all_scores, all_labels = [], []

            for i in range(N_ITERATIONS):
                
                ''' Computing final "training AUC" for this classifier '''
                ''' 5-Fold Cross Validation (without anymore gene filtering)''' 
                s, l = cv_kfold(X_big_train[sorted_best_genes[:n_final_genes]],
                                y_big_train, K_FOLDS, clf)
                all_scores += s
                all_labels += l
            
            ''' --------- Producing ROC ---------- '''
            print('Producing "training AUC" for this classifier')
            ''' Generating required data '''
            fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            ''' Plotting '''
            plt.title('ROC Curve for {} FS Algorithm'.format(fs_func))
            plt.plot(fpr, tpr, label = '%s - top %d genes, AUC = %0.2f' \
                     % (clf['name'], n_final_genes, roc_auc))
    
    ''' -------------------- DISPLAYING FINAL FIGURE ------------------------- '''    
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

''' Print the best genes used at each stage '''
print(log_best_genes)
    
''' ----------------------------------------------------------------------- '''
