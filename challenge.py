import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#read data
dataframe = pd.read_csv('challenge_dataset.txt',header=None,names=['colA','colB'])
x_values = dataframe[['colA']]
y_values = dataframe[['colB']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
y_predictions = body_reg.predict(x_values)
mse = mean_squared_error(y_values, y_predictions)
print(mse)
plt.plot(x_values, y_predictions)
plt.show()
