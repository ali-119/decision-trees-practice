import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# upload dataset
df = pd.read_csv(r'C:\Users\Banavand\Desktop\File\titanic\train.csv')

# print(df.isna().sum())
'''
    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
'''
# print(df.info())
'''
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype
    ---  ------       --------------  -----
     0   PassengerId  891 non-null    int64
     1   Survived     891 non-null    int64
     2   Pclass       891 non-null    int64
     3   Name         891 non-null    object
     4   Sex          891 non-null    object
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64
     7   Parch        891 non-null    int64
     8   Ticket       891 non-null    object
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object
     11  Embarked     889 non-null    object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
'''

# Organizing the dataset, adjusting for missing data and outliers
df = df.drop('Cabin', axis=1)

sns.boxplot(data=df, x='Embarked', y='Age', hue='Embarked')
plt.show()

df = df[df['Age'] < 62]
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# print(df['Embarked'].unique())  # ['S' 'C' 'Q']

# Data segmentation and training
X = pd.get_dummies(df.drop("Embarked", axis=1), drop_first=True)
Y = df['Embarked']
x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=5, test_size=0.3)

# Model training
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Calculating metrics
pred = model.predict(x_test)
print(f"accuracy score: {accuracy_score(y_test, pred)}")  # accuracy score: 0.8038277511961722
print(f"\nclassification_report:\n{classification_report(y_test, pred)}")
'''
    classification_report:
                  precision    recall  f1-score   support
    
               C       0.60      0.44      0.51        41
               Q       0.71      0.50      0.59        10
               S       0.84      0.92      0.88       158
    
        accuracy                           0.80       209
       macro avg       0.72      0.62      0.66       209
    weighted avg       0.79      0.80      0.79       209
'''

# Drawing a model tree diagram
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns)
plt.show()
