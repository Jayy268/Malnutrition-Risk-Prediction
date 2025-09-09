# Malnutrition-Risk-Prediction

This project predicts malnutrition risk using machine learning techniques and provides an interactive **Gradio web app** for real-time predictions. The goal is to support early detection and intervention in malnutrition cases.

![Gradio Demo](assets/gradio_demo.gif)

---

## 📊 Key Features
- **Data Preprocessing:** Cleaned and prepared 47,000+ samples for modelling.
- **Feature Engineering:** Selected and transformed features for optimal model performance.
- **Ensemble Model:** Built a Voting Classifier combining Random Forest, XGBoost, and KNN.
- **High Accuracy:** Achieved over **90% accuracy**, outperforming baseline models by 59%.
- **Interactive Deployment:** Deployed using Gradio for easy use by frontline healthcare workers.

## 📥 Download the Trained Model
The trained model file is available under the [GitHub Release](https://github.com/Jayy268/Malnutrition-Risk-Prediction/releases/tag/v1.0).  
Download it and place it inside the `models/` folder before running the notebook or launching the Gradio app.

## ▶️ Running the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/Jayy268/Malnutrition-Risk-Prediction.git
   cd malnutrition-prediction
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Download the trained model**
    From Releases
    Place it in the models/ folder.
4. **Run the Gradio app**
    ```bash
    python app/gradio_app.py


## 🛠 Tech Stack

**Languages:** Python  
**Libraries:** scikit-learn, XGBoost, joblib, pandas, numpy, matplotlib, seaborn  
**Deployment:** Gradio  
**Version Control:** Git + GitHub

## 📚 Lessons Learned

- **Importance of proper data cleaning and feature selection** in improving model performance.  
- **How ensemble methods can boost predictive accuracy** by combining the strengths of multiple models.  
- **Deploying ML models with user-friendly interfaces** to make them accessible to non-technical users.  





   
