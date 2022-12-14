import pandas
from sklearn import linear_model

# get the training data, fit the model and then calculate the score based on test set
df = pandas.read_csv("train_data.csv")
x = df[['x1', 'x2', 'x3']]
y = df['star']
regr = linear_model.LinearRegression()
regr.fit(x, y)

#


