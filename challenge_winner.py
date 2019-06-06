import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn

# visualize original data

df = pd.read_csv('challenge_dataset.txt', names=['X', 'Y'])
# sns.regplot(x='X', y='Y', data=df, fit_reg=False)
# plt.show()

# 2D Regression

from sklearn.model_selection import train_test_split
# before version 0.18 model selection is available in cross_validation
# from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = np.asarray(train_test_split(df['X'], df['Y'], test_size=0.1))

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train.values.reshape(-1,1), Y_train.values.reshape(-1,1))
print('Score: ', reg.score(X_test.values.reshape(-1,1), Y_test.values.reshape(-1,1)))

# Plot regression and visualize result
x_line = np.arange(5,25).reshape(-1,1)
sns.regplot(x=df['X'], y=df['Y'], data=df, fit_reg=False)
plt.plot(x_line, reg.predict(x_line))
plt.show()