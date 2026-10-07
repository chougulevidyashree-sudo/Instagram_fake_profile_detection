from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    features = [
        float(request.form['followers']),
        float(request.form['following']),
        float(request.form['posts']),
        float(request.form['engagement_rate']),
        float(request.form['bio_length']),
        float(request.form['has_profile_pic']),
        float(request.form['username_digits']),
        float(request.form['is_private']),
        float(request.form['external_url']),
        float(request.form['avg_likes']),
        float(request.form['avg_comments']),
        float(request.form['hashtags_count']),
        float(request.form['account_age']),
        float(request.form['spam_words'])
    ]

    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = 'FAKE ACCOUNT'
        risk = 'HIGH'
    else:
        result = 'REAL ACCOUNT'
        risk = 'LOW'

    return render_template('result.html',
                           prediction=result,
                           risk=risk)

if __name__ == '__main__':
    app.run(debug=True)