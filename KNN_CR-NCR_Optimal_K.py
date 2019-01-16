''' K-FOLD EXERCISE - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

''' scorer(estimator, X, y) callable function for cross val score '''
def scorer(estimator, X, y):
    
    probas = estimator.predict_proba(X)
    scores = [p[1] for p in probas]
    labels = y
    
    ''' Computing AUC '''
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    
    return auc

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
X_big_train, X_final_test, y_big_train, y_final_test = \
    model_selection.train_test_split(X_df, cr_ncr_df, test_size=0.20,
                                    random_state=42, stratify=cr_ncr_df)

''' Leaving only pval genes'''
print('getting significant genes')

from scipy.stats import ttest_ind

pval_genes = []

CR_samples = X_big_train.loc[[s for s in y_big_train.index if y_big_train.loc[s][0] == 'CR']]
not_CR_samples = X_big_train.loc[[s for s in y_big_train.index if y_big_train.loc[s][0] == 'Not in CR']]

for gene in X_big_train.columns:
    p_value = ttest_ind(CR_samples[gene], not_CR_samples[gene])[1]
    if (p_value <= 0.001):
        pval_genes.append(gene)
        
print('got the pval genes')

''' ---- Plotting Misclassification error to find optimal K for K-NN --- '''
# getting ground truth
ground_truth = [1 if y == 'CR' else 0 for y in y_big_train.values]

# Creating new (large) figure
plt.figure(figsize=(8, 6))

# creating list of K for KNN
neighbors = list(range(1,50))[::2]

# 2 optimal Ks (1 is with ttest feature selection)
two_best_Ks = []

for X in [X_big_train, X_big_train[pval_genes]]:
    # list of lists with AUCs of each K
    k_aucs = [[] for i in range(len(neighbors))]
    
    # For a 100 times
    for i in range(100):
        # empty list that will hold cv scores
        cv_scores = []
        # For each K perform 5-fold cv
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = model_selection.cross_val_score(knn, X, ground_truth,
                                                     cv=StratifiedKFold(n_splits=5, shuffle=True),
                                                     scoring=scorer)
            cv_scores.append(scores.mean())
            k_aucs[neighbors.index(k)].append(scores.mean())

        plt.plot(neighbors, cv_scores)
    
    k_scores = [np.mean(AUCs) for AUCs in k_aucs]
    # determining average of optimal Ks
    two_best_Ks.append(neighbors[k_scores.index(max(k_scores))])

# plot AUCs for different Ks
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.1))
plt.xlabel('Number of Neighbors K')
plt.ylabel('AUC Result')
plt.title('Avg of Optimal Ks: without ttest = %d (bottom); with ttest = %d (top)' 
          % (two_best_Ks[0], two_best_Ks[1]))
plt.show()
    
''' ----------------------------------------------------------------------- '''