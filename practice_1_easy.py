import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)

# print(df.head())
'''
            sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0            5.1               3.5                1.4               0.2
    1            4.9               3.0                1.4               0.2
    2            4.7               3.2                1.3               0.2
    3            4.6               3.1                1.5               0.2
    4            5.0               3.6                1.4               0.2
'''

X = df
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=13, test_size=0.3)

model = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=4)
model.fit(x_train, y_train)

pred = model.predict(x_test)

print(f"accuracy score: {accuracy_score(y_test, pred)}") # accuracy score: 0.9777777777777777

print(f"\nclassification_report:\n{classification_report(y_test, pred)}")
'''
classification_report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        14
           1       0.92      1.00      0.96        12
           2       1.00      0.95      0.97        19

    accuracy                           0.98        45
   macro avg       0.97      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
'''

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=df.columns)
plt.show()
