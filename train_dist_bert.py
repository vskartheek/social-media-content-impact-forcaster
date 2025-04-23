import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import DataLoader, Dataset
import joblib

# Load Data
data = pd.read_excel('social_media_engagement_data.xlsx', nrows=10000)

# Handle Missing Data
data.fillna({'Influencer ID': 'unknown'}, inplace=True)

# Define Target Variable (High Engagement = 1, Low Engagement = 0)
data['Engagement'] = (data['Engagement Rate'] > data['Engagement Rate'].median()).astype(int)

# Text Dataset for DistilBERT
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize DistilBERT Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Preprocess Text Data
texts = data['Post Content'].fillna('').tolist()
labels = data['Engagement'].tolist()

# Tokenize and Create Dataset
dataset = TextDataset(texts, labels, tokenizer, max_len=128)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Extract DistilBERT Features
def extract_distilbert_features(data_loader, model, device):
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            features.append(cls_embeddings.cpu().numpy())

    return np.vstack(features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
distilbert_model.to(device)
distilbert_features = extract_distilbert_features(data_loader, distilbert_model, device)

# Combine DistilBERT Features with Other Data
non_text_features = data[['Platform', 'Likes', 'Comments', 'Shares', 'Impressions', 'Reach']]
non_text_features = pd.get_dummies(non_text_features, columns=['Platform'], drop_first=True)

# Final Feature Matrix
X = np.hstack([distilbert_features, non_text_features.values])
y = data['Engagement'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save Model
joblib.dump(clf, 'engagement_model_distilbert.pkl')

# Evaluate Model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


from sklearn.metrics import accuracy_score

# Evaluate Model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Multi regressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Define Target Variables
targets = data[['Likes', 'Comments', 'Impressions', 'Reach']].values


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)

# Train Multi-Output Regressor
regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
regressor.fit(X_train, y_train)

# Save Model
joblib.dump(regressor, 'engagement_model_distil_multioutput.pkl')

# Evaluate Model
y_pred = regressor.predict(X_test)
print("Model evaluation completed.")


