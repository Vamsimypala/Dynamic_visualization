from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
data = pd.read_csv('final_hack\d.csv')


features = data[['Recipe_Complexity', 'Ingredients', 'Chef_Experience']]
target = data['Cooking_Time']


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


def predict_cooking_time(recipe_complexity, ingredients, chef_experience):
    
    predicted_cooking_time = model.predict([[recipe_complexity, ingredients, chef_experience]])
    return predicted_cooking_time[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    recipe_complexity = float(data['recipe_complexity'])
    ingredients = float(data['ingredients'])
    chef_experience = float(data['chef_experience'])

    
    predicted_cooking_time = predict_cooking_time(recipe_complexity, ingredients, chef_experience)

   
    response = {'predicted_cooking_time': predicted_cooking_time}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
