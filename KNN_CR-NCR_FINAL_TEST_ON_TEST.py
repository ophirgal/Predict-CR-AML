''' K-FOLD EXERCISE - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
                          index_col=[1]).T.iloc[:1].T

''' Extracting BALANCED target set from clinical_df (at 2nd CR category) '''

ncr_df = clinical_df.loc[clinical_df['CR status at end of course 2'] == 'Not in CR']
cr_df = clinical_df.loc[clinical_df['CR status at end of course 2'] == 'CR']
cr_ncr_df = cr_df.append(ncr_df)[['CR status at end of course 2']]
cr_ncr_df = cr_ncr_df.sort_index()

''' Extracting data set from df '''
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


''' setting best hill climbed genes '''

final_gene_names = ['MAT2B', 'MRPL32', 'POLR1C', 'RP11-408P14.1', 'WBP2',
                    'GBAS', 'RP11-934B9.3', 'RP11-113C12.4', 'AC090602.1',
                    'EZH1', 'UQCRFS1', 'CMTM3', 'CFL2', 'BTBD8', 'C18orf21']

final_genes = [gid for gid in gid_to_name.index
               if gid_to_name.loc[gid][0] in final_gene_names]


for k in list(range(1,37))[::2]:
    ''' Instantiating classifier '''
    knn = KNeighborsClassifier(k)
    
    knn.fit(X_big_train[final_genes],
            [1 if y == 'CR' else 0 for y in y_big_train.values])
    probas = knn.predict_proba(X_final_test[final_genes])
    
    scores = [p[1] for p in probas]
    labels = [1 if y == 'CR' else 0 for y in y_final_test.values]
    
    ''' --------- Producing ROC ---------- '''
    print('Producing "training AUC" for this classifier')
    ''' Generating required data '''
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    ''' Plotting '''
    plt.title('Final ROC Curve for K-Nearest Neighbors')
    plt.plot(fpr, tpr, label = 'K-NN: K = %d, Top 15 Genes, AUC = %0.2f' \
             % (k, roc_auc))




''' -------------------- DISPLAYING FINAL FIGURE ------------------------- '''    
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
    
''' ----------------------------------------------------------------------- '''