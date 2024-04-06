from flask import Flask, request, jsonify
from pymongo import MongoClient
import joblib

app = Flask(__name__)
model = joblib.load('decision_tree_model.pkl')

# Connect to MongoDB
client = MongoClient('mongodb+srv://pranjalbcse2283:123321@witty.iwn24tf.mongodb.net')
db = client['feed']  # Replace 'your_database_name' with your actual database name
feedback_collection = db['feedback']  # Collection for storing feedback

@app.route('/', methods=['GET'])
def hello():
    return "Hello from Flask!"

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    data = request.get_json()
    features = data['features']
    # Assuming you preprocess features if needed before making predictions
    prediction = model.predict([features])[0]
    # Convert int64 to regular int for JSON serialization
    prediction = int(prediction)
    return jsonify({'prediction': prediction})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    email = data.get('email')
    feedback = data.get('feedback')

    if email and feedback:
        # Save feedback to MongoDB
        feedback_document = {'email': email, 'feedback': feedback}
        feedback_collection.insert_one(feedback_document)
        return jsonify({'message': 'Feedback submitted successfully!'})
    else:
        return jsonify({'error': 'Email and feedback are required.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
