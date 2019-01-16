''' K-FOLD EXERCISE - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from scipy.stats import spearmanr

def get_gene_importances(X, y, folds, gene_to_importances, all_AUCs):
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        print('fold')
        ''' Get train and test data '''
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ''' fit and predict '''
        rf = RandomForestClassifier(n_estimators=N_TREES, 
                                    max_features=int(np.sqrt(len(X_train.columns))),
                                    max_depth=int(np.sqrt(len(X_train.columns))),
                                    random_state=42)
        rf.fit(X_train, [1 if y == 'CR' else 0 for y in y_train.values])
        fimp = list(rf.feature_importances_)
        cnt = 0
        for g in gene_to_importances:
            gene_to_importances[g].append(fimp[cnt])
            cnt += 1
        probas = rf.predict_proba(X_test)
        scores = [p[1] for p in probas]
        labels = [1 if y == 'CR' else 0 for y in y_test.values]
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        all_AUCs.append(roc_auc)

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

''' --------------- END OF FETCHING AND ORGANIZING DATA ------------------- '''

''' Important Constants '''
N_ITERATIONS = 100
K_FOLDS = 5

# Decide on Hyperparameters
''' Defining Hyperparameters '''
N_TREES = 100

'''--------  100 reps with diff splits - NOT SURE IF THIS IS A GOOD IDEA -- '''
plt.figure(figsize=(8, 6))

all_scores = {5:[], 10:[], 15:[], 20:[], 30:[], 50:[], 100:[]}
all_labels = {5:[], 10:[], 15:[], 20:[], 30:[], 50:[], 100:[]}

for i in range(N_ITERATIONS):

    print('BIG Iteration',i+1)
    
    # Leave 1/5 out (of training set)
    ''' Spliting training set '''
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = train_test_split(X_big_train,
                                                                    y_big_train,
                                                                    test_size=0.20,
                                                                    random_state=42 + i,
                                                                    stratify=y_big_train)
    
    # 5-Fold CV for N_ITERATIONS iterations (generate importance vector per gene)
    gene_to_importances = dict(zip(gid_to_name.columns,
                              [[] for i in range(len(gid_to_name.columns))]))
    
    all_AUCs = []
    
    for j in range(N_ITERATIONS):   
        
        print('CV Iteration', j+1)
        
        ''' 5-Fold CV ''' 
        get_gene_importances(X_sub_train, y_sub_train, K_FOLDS,
                             gene_to_importances, all_AUCs)
    
    # Sort genes by descending Spearman correlation between importance and the AUC(AUC of each fold)
    ''' Keeping only non-all-zero genes ([0,0,0,0,....]) in dict '''
    new_dict = {}
    
    for g in gene_to_importances:
        if np.mean(gene_to_importances[g]) != 0:
            new_dict[g] = gene_to_importances[g]
    
    gene_to_importances  = new_dict
    
    genes_by_importance = sorted(gene_to_importances, reverse=True,
                          key=lambda x: spearmanr(gene_to_importances[x], all_AUCs)[0])
    
    ''' Splitting the Training Set '''
    X_sub_train, X_sub_test, y_sub_train, y_sub_test = \
        train_test_split(X_big_train, y_big_train,
                         test_size=0.20, random_state=42 + i,
                         stratify=y_big_train)
    
    for n_final_genes in [5, 10, 15, 20, 30, 50, 100]:
        
        rf = RandomForestClassifier(n_estimators=N_TREES, 
                                    max_features=int(np.sqrt(n_final_genes)),
                                    max_depth=int(np.sqrt(n_final_genes)),
                                    random_state=42)
        rf.fit(X_sub_train[genes_by_importance[:n_final_genes]],
               [1 if y == 'CR' else 0 for y in y_sub_train.values])
        probas = rf.predict_proba(X_sub_test[genes_by_importance[:n_final_genes]])
        all_scores[n_final_genes] += [p[1] for p in probas]
        all_labels[n_final_genes] += [1 if y == 'CR' else 0 for y in y_sub_test.values]

for n_final_genes in [5, 10, 15, 20, 30, 50, 100]:
 
    print('Producing "training AUC" with', n_final_genes, 'final genes')
    
    ''' Generating required data '''
    fpr, tpr, thresholds = metrics.roc_curve(all_labels[n_final_genes],
                                             all_scores[n_final_genes], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ''' Plotting '''
    plt.plot(fpr, tpr, label = 'top %d genes, AUC = %0.2f' \
             % (n_final_genes, roc_auc))

''' -------------------- DISPLAYING FINAL FIGURE ------------------------- '''    
plt.title('ROC Curve for Random Forest (%d trees, complex FS, 100 reps)' % N_TREES)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

''' ----------------------------------------------------------------------- '''