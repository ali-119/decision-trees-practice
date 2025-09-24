import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_wine()

X = pd.DataFrame(data=data.data, columns=data.feature_names)
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42,
                                                     test_size=0.4, train_size=0.6)

model = DecisionTreeClassifier().fit(x_train, y_train)

df = pd.DataFrame({
    "columns": X.columns,
    "feature_importances_": model.feature_importances_
})
# print(df)

sns.barplot(data=df, x="columns", y="feature_importances_")
plt.xticks(rotation=35)
plt.show()
