# Exp-4-Multivariate-Analysis

# AIM

To read the given data and perform Multivariate Analysiss.

# Explanation

Multivariate analysis is based in observation and analysis of more than one statistical outcome variable at a time. In design and analysis, the technique is used to perform trade studies across multiple dimensions while taking into account the effects of all variables on the responses of interest.

# ALGORITHM

### STEP 1

Read the given Data

### STEP 2

Get the information about the data

### STEP 3

Handle missing values

### STEP 4

Perform basic descriptive statistics

### STEP 5

Visualize the data

# CODE

    import pandas as pd

    #Import the SuperStore.csv dataset

    df = pd.read_csv("SuperStore.csv")

    #Remove any missing values

    df.dropna(inplace=True)

    #Remove any outliers using z-score

    from scipy import stats

    import numpy as np

    z = np.abs(stats.zscore(df["Sales"]))

    df = df[(z < 3)]

    #Compute descriptive statistics for each variable

    print(df.describe())

    #Compute the correlation matrix

    corr_matrix = df.corr()

    #Visualize the correlation matrix as a heatmap

    import seaborn as sns

    import matplotlib.pyplot as plt

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)

    plt.show()

    #Standardize the data

    from sklearn.preprocessing import StandardScaler

    features = ['Sales', 'Postal Code']

    x = df.loc[:, features].values

    x = StandardScaler().fit_transform(x)

    #Perform PCA

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)

    #Create a new dataframe with the principal components

    df_pca = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

    #Perform k-means clustering

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=3, random_state=0).fit(df_pca)

    #Add the cluster labels to the dataframe

    df_pca['cluster'] = kmeans.labels_

    #Create a scatterplot of the first two principal components, colored by cluster

    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='cluster')

    plt.show()

    #Create a bar chart of sales by category

    sns.barplot(data=df, x='Category', y='Sales')

    plt.show()

    #Create a histogram of sales by region

    sns.histplot(data=df, x='Sales', hue='Region')

    plt.show()

# OUTPUT

![image](https://user-images.githubusercontent.com/91734840/230870345-c973f757-fd16-43df-8f3e-3ac7be340b3e.png)
![image](https://user-images.githubusercontent.com/91734840/230870396-86ae1db6-5540-4736-9f33-aba1b7f78ffe.png)
![image](https://user-images.githubusercontent.com/91734840/230870516-e3f6baff-4a07-4906-9990-99a40fdeee3d.png)

# RESULT

Thus, to read the given data and perform Multivariate Analysiss has been performed successfully.
