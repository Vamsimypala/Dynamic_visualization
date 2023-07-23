import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import io
import base64
import pymongo
app = Flask(__name__)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
data_collection = db["data_collection"]

@app.route('/')
def upload():
    return render_template('upload-excel.html')

@app.route('/view', methods=['POST'])
def view():
    file = request.files['file']
    file.save(file.filename)
    data = pd.read_excel(file)
    data.columns = data.columns.astype(str)
    data_list = data.to_dict(orient='records')
    data_collection.insert_many(data_list)

    plt.figure(figsize=(10, 5))
    plt.bar(data.iloc[:, 0], data.iloc[:, 1])
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Plot')

    bar_buffer = io.BytesIO()
    plt.savefig(bar_buffer, format='png')
    bar_buffer.seek(0)

    plt.clf()

    pie_charts = []

    for column in data.columns[:]:
        labels = data.iloc[:, 0].tolist()
        values = data[column].tolist()

        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        plt.title(f'Pie Chart for {column}')

        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        pie_buffer.seek(0)

        pie_plot_base64 = base64.b64encode(pie_buffer.getvalue()).decode()

        pie_charts.append((column, pie_plot_base64))

        plt.clf()

    return render_template('view-data.html', table=data.to_html(), bar_plot_base64=base64.b64encode(bar_buffer.getvalue()).decode(), pie_charts=pie_charts)
if __name__ == '__main__':
    app.run(debug=True)
