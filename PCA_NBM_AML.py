''' PCA of CR vs. NCR - Ophir Gal'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import preprocessing

''' ------------------------- MAIN PROGRAM -------------------------------- '''
''' Importing data while removing outlier data point (happens to be the 1st one)'''
print('importing AML_NBM')
df = pd.read_csv('C:/Users/user/Desktop/NBM_AML.txt',
                 delimiter='\t', header=[0], index_col=[1]).T.iloc[1:,:]

''' dropping duplicates in AML data (not keeping any)'''
df = df.reset_index(0).drop_duplicates('index', keep=False).set_index('index')

''' Fetching gene id to gene name mapping '''
gid_to_name = pd.read_csv('C:/Users/user/Desktop/NBM_AML.txt', delimiter='\t',
                          index_col=[1]).T.iloc[:1]

''' --------------- END OF FETCHING AND ORGANIZING DATA ------------------- '''
#########################
#
# Perform PCA on the data
#
#########################

# First center and scale the data
scaled_data = preprocessing.scale(df)
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
pca_df = pd.DataFrame(pca_data, index=df.index, columns=labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
    
ax.scatter(pca_df.PC1.loc[df.index[:20]],
           pca_df.PC2.loc[df.index[:20]],
           pca_df.PC3.loc[df.index[:20]],
           c='b', marker='o')

ax.scatter(pca_df.PC1.loc[df.index[20:]],
           pca_df.PC2.loc[df.index[20:]],
           pca_df.PC3.loc[df.index[20:]],
           c='r', marker='^')

ax.set_xlabel('PC1: {0}%'.format(per_var[0]))
ax.set_ylabel('PC2: {0}%'.format(per_var[1]))
ax.set_zlabel('PC3: {0}%'.format(per_var[2]))



for i in range(20):
    sample = pca_df.columns[i]    
    ax.text(pca_df.loc['PC1'].iloc[i],
            pca_df.loc['PC2'].iloc[i],
            pca_df.loc['PC3'].iloc[i], sample[:6])
    


plt.title('PCA: 20 Healthy Samples (blue) & 473 NCR (red)')

plt.show()

