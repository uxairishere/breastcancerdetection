import pandas as pd
data = pd.read_csv("dataset/dataset_stan.csv")
data.head()

inputs = data.drop('Classification', axis='columns')
target = data['Classification']
print(target)

inputs_ext = inputs.drop(['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1', 'Classification_st'], axis = 'columns')
print(inputs_ext)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_ext, target)

value = model.score(inputs_ext, target)
# print(value)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

tree = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.75, random_state=0)

tree = tree.fit(X_train,y_train)

y_predict = tree.predict(X_test)

print(metrics.confusion_matrix(y_test, y_predict))

print("Accuracy:", metrics.accuracy_score(y_test, y_predict))

result = model.predict([[1.788816, -0.080447, 1.792721018, 0.987394168, 1.125765748, 3.333167101, 0.576644063, -0.841036164, -1.290746828]])

if result == 2:
    print(result, "The Person is a patient of breast cancer")
else:
    print(result, "The Person is not a patient of breast cancer")
