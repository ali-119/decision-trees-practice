import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# loading data
df = pd.read_csv(r"...\File\penguins_size.csv")

# print(df.info())

'''
         Column             Non-Null Count   Dtype  
    ---  ------            --------------    -----  
     0   species            344 non-null    object 
     1   island             344 non-null    object 
     2   culmen_length_mm   342 non-null    float64
     3   culmen_depth_mm    342 non-null    float64
     4   flipper_length_mm  342 non-null    float64
     5   body_mass_g        342 non-null    float64
     6   sex                334 non-null    object 
'''


# Control of missing values
# print(df.isna().sum())

'''
    species               0
    island                0
    culmen_length_mm      2
    culmen_depth_mm       2
    flipper_length_mm     2
    body_mass_g           2
    sex                  10
    dtype: int64
'''

df = df.dropna()
# print(df.info())

# print(df['sex'].unique())
# print(df[df['sex'] == '.'])

# print(df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose())

a = df.at[336, 'sex'] = 'FEMALE'
b = df.loc[336]


# Observing the degree of dependence, correlation, and relationship between values
# sns.scatterplot(data=df, x='culmen_length_mm', y='culmen_depth_mm',
#                  hue='species', palette='Dark2')

# sns.pairplot(data=df, hue='species', palette='Dark2')

# sns.catplot(data=df, x='species', y='culmen_length_mm',
#              kind='box', col='sex', hue='species', palette='Dark2')

# plt.show()


# Data segmentation
x = pd.get_dummies(df.drop("species", axis=1), drop_first=True)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=101, test_size=0.3)


# Model training
max_depth = [2, 5, 10]
max_leaf_nodes = [3, 5, 10]

for depth in max_depth:
    for leaf in max_leaf_nodes:
        model = DecisionTreeClassifier(max_depth=depth, max_leaf_nodes=leaf)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)

        print(f"\n\naccuracy score for max_depth {depth} and max_leaf_nodes {leaf}: {accuracy_score(y_test, pred) * 100:.2f}%")
        print(f"\nclassification_report for max_depth {depth} and max_leaf_nodes {leaf}:\n{classification_report(y_test, pred)}")

        plt.figure(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=x.columns)
        plt.title(f"for max_depth {depth} and max_leaf_nodes {leaf}", fontsize=20)
        plt.show()
