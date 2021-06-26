import pandas as pd
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dataset/dataset_stan.csv")

Raw_data = data.head()
print(Raw_data)

# Now, set the independent variables (represented as X) and the dependent variable (represented as y):
# Here X (Independent variable) = inputs_ext
# And  Y (dependent variable)   = target

inputs = data.drop('Classification', axis='columns')
target = data['Classification']
print(target)

inputs_ext = inputs.drop(['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1', 'Classification_st'], axis = 'columns')
print(inputs_ext)

x_train, x_test, y_train, y_test = train_test_split(inputs_ext, target, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))