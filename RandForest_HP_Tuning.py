''' K-FOLD EXERCISE - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

''' Important Constants '''
N_ITERATIONS = 100
K_FOLDS = 5
N_TREES = 50

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
'''from sklearn.preprocessing import normalize
X_df = pd.DataFrame(data=normalize(X_df, norm='max'), columns=X_df.columns,
                    index=X_df.index)
'''
''' IMPORTANT - CAREFUL - KEEEP random_state equal to 42 (meaning of life) '''
''' Splitting into train and test subsets (setting test set aside till Poster) '''
X_big_train, X_final_test, y_big_train, y_final_test = train_test_split(X_df,
                                                                cr_ncr_df,
                                                                test_size=0.20,
                                                                random_state=42,
                                                                stratify=cr_ncr_df)

''' --------------- END OF FETCHING AND ORGANIZING DATA ------------------- '''
AUCs = []
plt.figure(figsize=(8, 6))

for p in [0.6, 0.8, 1, 1.2, 1.4]:

    print('p =', p)
    
    for n_trees in [10, 25, 50 ,75, 100]:
        
        all_labels, all_scores = [], []
        
        for i in range(N_ITERATIONS):
        
            print('Iteration',i+1)
            
            ''' Splitting the Training Set '''
            X_sub_train, X_sub_test, y_sub_train, y_sub_test = \
                train_test_split(X_big_train, y_big_train,
                                 test_size=0.20, random_state=42 + n_trees + i,
                                 stratify=y_big_train)
            
            ''' Getting Classifier '''
            rf = RandomForestClassifier(n_estimators=n_trees, 
                                        max_features=int(p*np.sqrt(len(X_sub_train.columns))),
                                        max_depth=int(p*np.sqrt(len(X_sub_train.columns))),
                                        random_state=42 + n_trees + i)
            rf.fit(X_sub_train, [1 if y == 'CR' else 0 for y in y_sub_train.values])
            probas = rf.predict_proba(X_sub_test)
            all_scores += [p[1] for p in probas]
            all_labels += [1 if y == 'CR' else 0 for y in y_sub_test.values]
        
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        ''' Plotting '''
        plt.plot(fpr, tpr, label = 'p = %0.1f, %d Trees, AUC = %0.2f' \
                 % (p, n_trees, roc_auc))
        AUCs.append(roc_auc)

''' -------------------- DISPLAYING FINAL FIGURE ------------------------- '''    
plt.title('ROC Curve - Random Forest (Different Hyperparameters)')
plt.legend(loc='lower right', fontsize=8)
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
    
''' ----------------------------------------------------------------------- '''