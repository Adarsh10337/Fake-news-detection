from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

app = Flask(__name__)

# Load the datasets
true_data = pd.read_csv('True (1).csv')
fake_data = pd.read_csv('Fake (1).csv')

# Preprocess the datasets
vectorizer = CountVectorizer()
X_true = vectorizer.fit_transform(true_data['title'])
X_fake = vectorizer.transform(fake_data['title'])

# Prepare the labels
y_true = [1] * X_true.shape[0]
y_fake = [0] * X_fake.shape[0]

# Combine the data and labels
X = vstack([X_true, X_fake])
y = y_true + y_fake

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Define the routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    sentence = request.form['sentence']
    X_sentence = vectorizer.transform([sentence])
    prediction = model.predict(X_sentence)[0]
    if prediction == 1:
        result_text = 'TRUE'
    else:
        result_text = 'FAKE'
    return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
