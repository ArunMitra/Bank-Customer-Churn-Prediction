from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Form page to submit text
@app.route('/')
def submission_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'] )
def predict():
    user_data = request.json

    with open("../../Model/model.pkl", 'rb') as f:

        model = pickle.load(f)
        X = np.array([user_data['credit_score'],
                      user_data['gender'],
                      user_data['age'],
                      user_data['tenure'],
                      user_data['balance'],
                      user_data['products_number'],
                      user_data['credit_card'],
                      user_data['active_member'],
                      user_data['estimated_salary'],
                      user_data['country_France'],
                      user_data['country_Germany']]).astype(float).reshape(1, -1)
        probs = model.predict_proba(X)
        print(probs)
        return jsonify({'WillChurn': probs[0][0],
                        'WontChurn': probs[0][1]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
