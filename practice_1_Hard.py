import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_wine()

X = pd.DataFrame(data=data.data, columns=data.feature_names)
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42,
                                                     test_size=0.4, train_size=0.6)

param_grid = {
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'max_depth': [3, 5, 7],
    'criterion': ['entropy', 'gini', 'log_loss']
}

model = DecisionTreeClassifier()

grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=1, cv=3)
grid.fit(x_train, y_train)

pred = grid.predict(x_test)

print(f"accuracy: {accuracy_score(y_test, pred) * 100:.2f}%\n")  # accuracy: 93.06%
print(f"best param: {grid.best_params_}")
# best param: {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 4}

plt.figure(figsize=(11, 9))
plot_tree(grid.best_estimator_, feature_names=X.columns, filled=True)
plt.title("best estimator tree plot", fontsize=16, family='Arial')
plt.show()
