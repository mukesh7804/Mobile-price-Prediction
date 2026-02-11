#  Mobile Price Range Prediction

This machine learning project predicts the price range of a mobile phone based on various technical specifications. It was built as part of an internship project and includes full model training, evaluation, and deployment using Flask.



##  Problem Statement

The goal is to predict the price category (range) of a mobile device given its features such as RAM, battery power, screen size, etc. The target variable `price_range` is a multi-class label with four categories (0 to 3).


##  Demo Video

Click to watch a short demo of the working project interface:

[Watch Demo 1](https://drive.google.com/file/d/19Y4xytJNVtI2acNoo1MOrT3UkeAhn4Nd/view?usp=sharing)
[Watch Demo 2](https://drive.google.com/file/d/1mvo7wYxkg2ZRTvqGEh5_hpM7eNhZt28B/view?usp=sharing)
> 💡 _Not deployed online — this video demonstrates how the app works locally._


##  Screenshots

> Below are screenshots of the deployed model:

- **Input Form** (HTML UI with feature fields)<br><br>
  ![front-end](scrrenshots/front_end1.png)
  
  ![front-end](scrrenshots/front_end2.png)<br><br><br>

  
- **Prediction Output** (Displayed after form submission)<br><br>
  ![first_input](scrrenshots/input1.png)<br>
  ![first_input](scrrenshots/output1.png)
---
  <br><br>
  ![first_input](scrrenshots/input2.png)<br>
  ![first_input](scrrenshots/output2.png)
---
  <br><br>
    ![first_input](scrrenshots/input3.png)<br>
  ![first_input](scrrenshots/output3.png)
---
  <br><br><br>
- **Confusion Matrix** (From evaluation notebook)<br>
  ![first_input](scrrenshots/confusion_matrix.png)<br>





##  Dataset Overview

- **Source:** Provided by Unified Mentor (CSV format)
- **Total Records:** 2000
- **Target Variable:** `price_range` (0: Low, 1, 2, 3: High)
- **Feature Types:** Mix of numerical and binary categorical features
- **Additional Features Created:**
  - `build_score` = `mobile_wt / m_dep`
  - `px_area` = `px_height * px_width`
  - `screen_area` = `sc_h * sc_w`



##  Data Preprocessing

- Verified no null values
- Engineered new features and dropped unused columns
- Applied `StandardScaler` on selected numerical features
- Binary features were passed as-is
- Used `train_test_split` (80/20) with stratification
- Built a full preprocessing + classification pipeline using scikit-learn



##  Model Building

- **Model Used:** Logistic Regression (`max_iter=1000`, `solver='lbfgs'`)
- **Pipeline:** Combined preprocessing and model into one scikit-learn pipeline
- **Model File Saved:** `mobile_price_model.pkl` using Joblib



## Model Evaluation

- **Accuracy:** 93.75%
- **Classification Report:**
  - Precision, Recall, and F1-scores were all high
  - Best performance on class 0 and 3; slightly lower on class 2
- **Confusion Matrix:** Visualized to confirm minimal misclassifications



##  Web Deployment

- Built a web interface using **Flask** and **HTML**
- User inputs mobile specs via a form on `index.html`
- On submission, the trained model predicts and displays the price range
- Flask backend reads user input, loads the model, and returns prediction
- All development and integration was done in **PyCharm**


##  Project Structure
-  `mobile_price_model.pkl` # Trained model
-  `index.html` # Frontend form
-  `app.py` # Flask backend
- `index.html` # HTML template
- ` requirements.txt` # Python dependencies
-  `README.md` # Project overview
-  `Cleaned_mobile_price_data.csv` #Cleaned dataset
-  `Phone_price_predictor.ipynb` #model training


##  Learnings

- Built complete ML pipeline from scratch
- Learned about data preprocessing, feature scaling, pipeline integration
- Implemented real-time model interaction using Flask
- Understood the end-to-end cycle of model development and deployment


##  Future Improvements

- Reduce number of inputs by selecting only high-impact features
- Improve form design and responsiveness
- Add more advanced models and perform hyperparameter tuning
- Host the application publicly (e.g., using Render or Heroku)
