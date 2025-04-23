from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the models
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the prediction models
bert_predictor = joblib.load("engagement_model_multioutput.pkl")
distilbert_predictor = joblib.load("engagement_model_distil_multioutput.pkl")
bert_classifier = joblib.load("engagement_model_bert.pkl")
distilbert_classifier = joblib.load("engagement_model_distilbert.pkl")

def extract_features(text, model_type='bert', device='cpu', max_len=128):
    if model_type == 'bert':
        model = bert_model
        tokenizer = bert_tokenizer
    else:
        model = distilbert_model
        tokenizer = distilbert_tokenizer
    
    model.eval()
    inputs = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        post_content = data.get("Post Content", "")
        platform = data.get("Platform", "")
        model_type = data.get("Model", "bert").lower()

        if not post_content or not platform:
            return jsonify({"error": "Invalid input. Both 'Post Content' and 'Platform' are required."}), 400

        # Extract features based on model type
        features = extract_features(post_content, model_type)

        # One-hot encode platform
        platforms = ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'YouTube']
        platform_features = [1 if platform == p else 0 for p in platforms]
        
        # Combine features
        platform_features = np.array(platform_features).reshape(1, -1)
        combined_features = np.hstack([features, platform_features])

        # Select models based on type
        regression_model = bert_predictor if model_type == 'bert' else distilbert_predictor
        classification_model = bert_classifier if model_type == 'bert' else distilbert_classifier
        
        # Ensure feature count matches for regression
        reg_expected_features = regression_model.n_features_in_
        current_features = combined_features.shape[1]

        if current_features < reg_expected_features:
            padding = np.zeros((1, reg_expected_features - current_features))
            reg_features = np.hstack([combined_features, padding])
        elif current_features > reg_expected_features:
            reg_features = combined_features[:, :reg_expected_features]
        else:
            reg_features = combined_features

        # Ensure feature count matches for classification
        class_expected_features = classification_model.n_features_in_
        if current_features < class_expected_features:
            padding = np.zeros((1, class_expected_features - current_features))
            class_features = np.hstack([combined_features, padding])
        elif current_features > class_expected_features:
            class_features = combined_features[:, :class_expected_features]
        else:
            class_features = combined_features

        # Make predictions
        regression_pred = regression_model.predict(reg_features)
        classification_pred = classification_model.predict(class_features)

        return jsonify({
            "Likes": int(regression_pred[0][0]),
            "Comments": int(regression_pred[0][1]),
            "Impressions": int(regression_pred[0][2]),
            "Reach": int(regression_pred[0][3]),
            "ReachClassification": int(classification_pred[0]),
            "ReachStatus": "Reach" if classification_pred[0] == 1 else "No Reach"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)