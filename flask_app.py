from flask import Flask, request, render_template
from sklearn import linear_model
import pandas as pd
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Загружаем и подготавливаем данные
df = pd.read_csv('/Users/Iana/Desktop/Diploma/data/olist_customers_dataset.csv')
y = df['review_score'] 
X = df[['price', 'product_weight_g']]

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Обучаем модель
lm = linear_model.LinearRegression()
lm.fit(X, y)

@app.route("/")
def home():
    return render_template('index.html', error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        price = float(request.form.get("Price", 0))
        product_weight_g = float(request.form.get("Product_weight_g", 0))

    except (ValueError, KeyError):
        return render_template('index.html', error='Invalid input data'), 400
        
    if price <= 0 or product_weight_g <= 0:
        return render_template('index.html', error='Price and Product weight must be positive numbers'), 400

    try:
        prediction = lm.predict([[price, product_weight_g]])
        output = round(prediction[0], 2)
    except Exception as e:
        return render_template('index.html', error=str(e)), 500

    return render_template('index.html', prediction_text=f'Товар по цене {price} с весом {product_weight_g} получит оценку {output}.', error=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)