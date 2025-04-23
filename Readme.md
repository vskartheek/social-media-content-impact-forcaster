    Social Media Engagement Prediction

This project predicts the engagement (likes/comments) a social media post might receive based on its caption and metadata using BERT and DistilBERT models. It provides both a training pipeline and a web interface for prediction.

---------------------------------------

📌 Project Overview

Goal: Predict post engagement using transformer-based models.

Models Used: BERT, DistilBERT (Single & Multi-output versions)

Interface: Flask-based web application

Input: Captions from social media posts

Output: Predicted number of likes and comments

---------------------------------------

🛠️ Requirements

Install required libraries using pip:

pip install -r requirements.txt

If requirements.txt is not available, you can manually install:

pip install flask pandas scikit-learn transformers torch openpyxl

---------------------------------------

🗂️ Project Structure

social_media_engagement/
    ├── app.py                    # Main Flask app
    ├── app_ext.py                # Alternate app (extended)
    ├── train_bert.py             # Script to train BERT model
    ├── train_dist_bert.py        # Script to train DistilBERT model
    ├── engagement_model_*.pkl    # Pre-trained models
    ├── social_media_engagement_data.xlsx  # Dataset
    ├── templates/
    │   └── index.html            # HTML template for UI
    └── Untitled.ipynb            # Experimentation notebook

---------------------------------------

🚀 How to Run the Project

1. Clone and Navigate

cd social_media_engagement

2. Ensure all dependencies are installed

3. train train_bert.py

        python train_bert.py

4. train dist_bert.py

        python train_dist_bert.py

--------------------------------------------------
Make sure the code is exited successfully and 
all the pkl files are generated
--------------------------------------------------

5. Run the Flask App

        flask --app app run

The app will run on http://127.0.0.1:5000/

6. Open in Browser

Navigate to http://127.0.0.1:5000/ and input a sample caption to get predictions.

---------------------------------------

🔄 To Train the Model (Optional)

You can retrain models using:

python train_bert.py
# or
python train_dist_bert.py

These scripts will train the model and output .pkl files which can be used by app.py

---------------------------------------

📊 Sample Input/Output

Input: "Enjoying the sunset on the beach!"
Platform: Instagram
AI Model: BERT
Output: Predicted Likes:- 12, Predicted Comments:- 0, Impressions: 1008, Reach:508
Reach Status: Reach

---------------------------------------

🤖 Models Used

BERT and DistilBERT from Hugging Face's transformers library

Fine-tuned on provided dataset for regression (likes/comments)

---------------------------------------

📁 Dataset

File: social_media_engagement_data.xlsx

Contains columns like: Post content, likes, comments, platform, shares, impressions, reach, engagement, etc.

---------------------------------------

🙌 Credits

Developed as part of a major project on social media analytics using AI.

Model weights are pre-trained and included for immediate testing.

---------------------------------------

📝 Notes

Make sure model .pkl files are in the same directory as app.py

