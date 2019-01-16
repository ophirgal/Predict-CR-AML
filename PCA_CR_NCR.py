''' PCA of CR vs. NCR - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing

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
#cr_ncr_df = cr_df.iloc[:len(ncr_df)].append(ncr_df)[['CR status at end of course 2']]
cr_ncr_df = cr_df.append(ncr_df)[['CR status at end of course 2']]
cr_ncr_df = cr_ncr_df.sort_index()

''' Extracting BALANCED data set from df '''
X_df = df.loc[cr_ncr_df.index]

''' --------------- END OF FETCHING AND ORGANIZING DATA ------------------- '''

#########################
#
# Perform PCA on the data
#
#########################

# First center and scale the data
scaled_data = preprocessing.scale(X_df)
pca = PCA() # create a PCA object
pca_data = pca.fit_transform(scaled_data) # do the math
 
#########################
#
# Draw a scree plot and a PCA plot
#
#########################

#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

#the following code makes a fancy looking plot using PC1 and PC2 (and PC3)
pca_df = pd.DataFrame(pca_data, index=X_df.index, columns=labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
    
ax.scatter(pca_df.PC1.loc[cr_df.index],
           pca_df.PC2.loc[cr_df.index],
           pca_df.PC3.loc[cr_df.index],
           c='b', marker='o')

ax.scatter(pca_df.PC1.loc[ncr_df.index],
           pca_df.PC2.loc[ncr_df.index],
           pca_df.PC3.loc[ncr_df.index],
           c='r', marker='^')

''' Anotating Year of Diagnosis '''
for i in range(len(pca_df)):
    text = clinical_df['CR status at end of course 2'][pca_df.index.values[i]]
    ax.text(pca_df.PC1.iloc[i],
            pca_df.PC2.iloc[i],
            pca_df.PC3.iloc[i], text, fontsize=8)


ax.set_xlabel('PC1: {0}%'.format(per_var[0]))
ax.set_ylabel('PC2: {0}%'.format(per_var[1]))
ax.set_zlabel('PC3: {0}%'.format(per_var[2]))

plt.title('PCA: 57 CRs (blue) & 57 NCR (red)')

plt.show()

