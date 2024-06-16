from sklearn import linear_model
import pandas as pd
import pickle

df = pd.read_csv('/Users/Iana/Desktop/Diploma/data/olist_customers_dataset.csv')

y = df['review_score'] # dependent variable
X = df[['price', 'product_weight_g']] # independent variable

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Заполняет пропущенные значения средним
X = imputer.fit_transform(X)

lm = linear_model.LinearRegression()
lm.fit(X, y) # fitting the model
pickle.dump(lm, open('model.pkl','wb')) # save the model

print(lm.predict([[30, 500]]))  # format of input
#print(f'score: {lm.score(X, y)}')