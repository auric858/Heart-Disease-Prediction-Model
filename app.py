import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

with open("heart_disease_scaler.pkl","rb") as f1:
    scaler=pickle.load(f1)
with open("heart_disease_pred_model.pkl","rb") as f2:
    model=pickle.load(f2)
st.set_page_config(page_title="Heart Disease Prediction ", layout="wide")
image=Image.open("image.jpg")
image2=Image.open("roc_curve.png")
df=pd.read_csv("heart.csv")


menu = ["🏠 Home", "📊 Data Overview","📉 Data Visualization","🧠 Prediction", "📈 Model Performance", "ℹ️ About"]
choice = st.sidebar.selectbox("Navigation", menu)
if choice=="🏠 Home":
    st.title("🫀 Heart Disease Prediction ")
    st.image(image,width="content")
    st.markdown("> A Machine Learning approach to predict heart disease risk")
    st.markdown("---")
    st.markdown("""
    This web app uses **Logistic Regression** to predict the likelihood of heart disease 
    based on key medical attributes such as age, cholesterol, blood pressure, and more.  
    It aims to support early risk detection and awareness through data-driven insights.
    """)
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Explore data insights in the **Data Analysis** section.  
    2. Enter patient details under **Prediction** to get the model output.  
    3. View overall **Model Performance** and evaluation metrics.  
    """)
    st.markdown("---")
    st.markdown("👨‍💻 **Developed by Auric Dutt** | 📊 Machine Learning Project")

elif choice=="📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("A quick look at the dataset used for predicting heart disease.")
    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df.head())
    st.markdown(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe().T)
    st.markdown("""
    >### **Dataset Summary**

    >1. The dataset contains **303 patient records** with **14 clinical features** used to predict heart disease.  
    >2. The **average patient age** is about **54 years**, ranging from **29 to 77**, and around **68% are male**.  
    >3. The **average resting blood pressure** is approximately **132 mmHg**, while the **mean cholesterol level** is **246 mg/dl**.  
    >4. Patients achieve an average **maximum heart rate of 150 bpm**, and the **oldpeak (ST depression)** typically lies around **1.0**, 
    with a few higher outliers up to **6.2**.  
    >5. About **32% of the patients experienced exercise-induced angina**, and roughly **54%** were diagnosed with 
    heart disease (**target = 1**), indicating a fairly balanced dataset suitable for binary classification.
    """)
    st.markdown("---")
    st.subheader("Data Types and Missing Values")
    data_info = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(data_info)
    st.markdown("""
    >### **Dataset Information**

    >The dataset contains **303 entries** with **14 attributes**, all of which are **complete (no missing values)**.  
    >Most of the columns are of **integer type (int64)**, except for *oldpeak*, which is a **floating-point feature (float64)**.
    """)
    st.markdown("---")
    st.subheader("Feature Description")
    st.markdown("""
    > **1. age:** Age in years  
    >
    > **2. sex:** Sex  
    > - 1 = Male  
    > - 0 = Female  
    >
    > **3. cp:** Chest pain type  
    > - 0 = Typical angina  
    > - 1 = Atypical angina  
    > - 2 = Non-anginal pain  
    > - 3 = Asymptomatic  
    >
    > **4. trestbps:** Resting blood pressure (in mm Hg on admission to the hospital)  
    >
    > **5. chol:** Serum cholesterol in mg/dl  
    >
    > **6. fbs:** (Fasting blood sugar > 120 mg/dl)  
    > - 1 = True  
    > - 0 = False  
    >
    > **7. restecg:** Resting electrocardiographic results  
    > - 0 = Normal  
    > - 1 = ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)  
    > - 2 = Probable or definite left ventricular hypertrophy (by Estes' criteria)  
    >
    > **8. thalach:** Maximum heart rate achieved  
    >
    > **9. exang:** Exercise induced angina  
    > - 1 = Yes  
    > - 0 = No  
    >
    > **10. oldpeak:** ST depression induced by exercise relative to rest  
    >
    > **11. slope:** The slope of the peak exercise ST segment  
    > - 0 = Upsloping  
    > - 1 = Flat  
    > - 2 = Downsloping  
    >
    > **12. ca:** Number of major vessels (0–3) colored by fluoroscopy  
    >
    > **13. thal:** Thalassemia type  
    > - 0 = Error (original dataset value mapped from NaN)  
    > - 1 = Fixed defect  
    > - 2 = Normal  
    > - 3 = Reversible defect  
    >
    > **14. target (label):**  
    > - 0 = No heart disease  
    > - 1 = Heart disease
    """)

elif choice=="📉 Data Visualization":
    st.title("📉 Data Visualization")
    st.markdown("""
        This section visualizes different patterns in the data to help understand how various health factors affect the chances of having heart disease.
        """)
    st.markdown("---")

    st.title("Heart Disease Distribution")
    fig,ax=plt.subplots(figsize=(2.5, 2))
    sns.countplot(x='target', data=df,ax=ax)
    ax.set_title('Heart Disease Distribution (0 = No, 1 = Yes)')
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    st.pyplot(fig,use_container_width=False)
    st.markdown("""
    ### **Target Variable Distribution**

    > This bar chart shows how many people in the dataset have heart disease (**1**) compared to those who don’t (**0**).  
    >
    > We can see that the number of patients **with heart disease** is slightly higher than those **without it**.  
    >
    > This indicates that the dataset is **fairly balanced**, which is good for model training as it helps prevent **bias toward one class**.
    """)
    st.markdown("---")
    st.title("Distribution of Numerical Features")
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for i, col in enumerate(num_features):
        sns.histplot(df[col], bins=15, kde=True, color='skyblue', ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig,use_container_width=False)
    st.markdown("""
    ### **Distribution of Numerical Features**

    > This set of histograms shows the distribution of key numerical features in the dataset —  
    > **age**, **trestbps** (resting blood pressure), **chol** (cholesterol), **thalach** (maximum heart rate), and **oldpeak** (ST depression).  
    >
    > **Age:** Most patients are between **45 and 60 years old**, showing that middle-aged people are more common in the dataset.  
    >
    > **Resting Blood Pressure (trestbps):** Mostly centered around **120–140 mmHg**, which is close to the normal range.  
    >
    > **Cholesterol (chol):** Slightly **right-skewed** — a few patients have very high cholesterol levels.  
    >
    > **Maximum Heart Rate (thalach):** Mostly between **130–170 bpm**, showing a normal healthy range.  
    >
    > **Oldpeak:** Highly **right-skewed**, meaning most patients have low ST depression, but a few show significantly higher values.
    """)
    st.markdown("---")
    st.title("Relationship between Numerical Features and Target")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, col in enumerate(['age', 'trestbps', 'chol', 'thalach', 'oldpeak']):
        row=i//3
        col_idx=i%3
        sns.boxplot(x='target', y=col, data=df, ax=axes[row,col_idx])
        axes[row,col_idx].set_title(f'{col} vs Target')
    axes[1,2].set_visible(False)
    st.pyplot(fig,use_container_width=False)
    st.markdown("""
    ### **Boxplot Analysis of Key Features vs Target**

    > The boxplots above show the relationship between some important health indicators and the **target variable** (presence or absence of heart disease).  
    >
    > - **Age vs Target:**  
    >   People with heart disease (**target = 1**) tend to be slightly **younger** on average compared to those without heart disease (**target = 0**).  
    >
    > - **Resting Blood Pressure (trestbps) vs Target:**  
    >   Both groups have a **similar distribution** of resting blood pressure, though patients without heart disease show **slightly higher variation**.  
    >
    > - **Cholesterol (chol) vs Target:**  
    >   Cholesterol levels are quite spread out for both groups, but the **median values are similar**. A few extreme outliers indicate **very high cholesterol** in some cases.  
    >
    > - **Maximum Heart Rate (thalach) vs Target:**  
    >   People with heart disease (**target = 1**) generally have **higher maximum heart rates**, suggesting that **healthier hearts reach higher rates** during exercise.  
    >
    > - **Oldpeak vs Target:**  
    >   **Oldpeak** (ST depression) is **higher for patients without heart disease**, while **lower oldpeak values** are often seen in those with heart disease.  
    >
    > ### **Summary:**  
    > Overall, patients with heart disease tend to be **younger**, have **higher maximum heart rates**, and **lower oldpeak values**, while other factors like cholesterol and resting blood pressure show less distinction.
    """)
    st.markdown("---")
    st.title("Categorical Features vs Target")
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    fig, axes = plt.subplots(3, 3, figsize=(13,8))
    for i, col in enumerate(cat_features):
        if i < len(cat_features):
            sns.countplot(x=col, hue='target', data=df, ax=axes[i // 3, i % 3])
            axes[i // 3, i % 3].set_title(f'{col} vs Target')
    plt.tight_layout()
    axes[2, 2].set_visible(False)
    st.pyplot(fig,use_container_width=False)
    st.markdown("""
    ## **Categorical Features vs Target**

    > These bar plots show how different **categorical features** relate to the presence of heart disease.  
    >
    > - **Sex:** More **females (sex = 0)** have heart disease compared to males (**sex = 1**).  
    > - **Chest Pain Type (cp):** Higher chest pain types (**2 and 3**) are linked with **more heart disease cases**.  
    > - **Fasting Blood Sugar (fbs):** Shows **very little difference** between the two groups.  
    > - **Resting ECG (restecg):** Slightly more heart disease cases occur when **restecg = 1**.  
    > - **Exercise Angina (exang):** People **without exercise-induced angina (exang = 0)** have more heart disease.  
    > - **Slope:** A **slope value of 2** is more common among patients with heart disease.  
    > - **Ca:** Fewer major vessels (**ca = 0**) are associated with a higher chance of heart disease.  
    > - **Thal:** **Thal value = 2** is most frequent among patients with heart disease.  
    >
    > ### **Summary:**  
    > **Chest pain type**, **slope**, and **exercise angina** show the clearest differences between healthy and heart disease groups.
    """)
    st.markdown("---")
    st.title("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Heart Disease Features")
    st.pyplot(fig,use_container_width=False)
    st.markdown("""
    ## **Correlation Heatmap Analysis**

    > The heatmap shows how different features relate to **heart disease**.  
    >
    > - **cp (chest pain type)** and **thalach (maximum heart rate)** are **positively correlated** with heart disease.  
    > - **oldpeak**, **exang (exercise angina)**, and **sex** are **negatively correlated**, meaning **higher values reduce the likelihood** of heart disease.  
    > - **chol (cholesterol)** and **trestbps (resting blood pressure)** show **weak correlation**, so they don’t have much direct impact.  
    >
    > ### **Summary:**  
    > **Chest pain type**, **heart rate**, **oldpeak**, and **exercise angina** are the **most influential features** associated with heart disease.
    """)

elif choice=="🧠 Prediction":
    st.title("🫀 Heart Disease Risk Prediction")
    st.markdown("---")
    st.markdown("""
    ### Enter Patient Details
    >Use the fields below to input patient health parameters and **predict the likelihood of heart disease**.  
    >This interactive form helps visualize how clinical features contribute to model-based predictions.
    """)
    #st.write("\n")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("**Age**", min_value=1, max_value=120, step=1)
        sex = st.radio("**Sex:**", ("**Male**", "**Female**"))
        sex = 1 if sex == "**Male**" else 0
        cp = st.selectbox("**Chest Pain Type (0-3)**", [0, 1, 2, 3])
        trestbps = st.slider("**Resting Blood Pressure(mm Hg)**", min_value=50, max_value=250,value=150)
        chol = st.slider("**Cholesterol(mg/dl)**", min_value=100, max_value=600, step=1)
        fbs=st.radio("**Fasting Blood Sugar > 120 mg/dl:**",("**Yes**","**No**"))
        fbs = 1 if fbs == "**Yes**" else 0
        restecg = st.selectbox("**Resting ECG (0-2)**", [0, 1, 2])
    with col2:
        thalach = st.number_input("**Maximum Heart Rate Achieved**", min_value=50, max_value=250, step=1,value=160)
        exang=st.radio("**Exercise Induced Angina:**",("**Yes**","**No**"))
        exang=1 if exang=="**Yes**" else 0
        oldpeak = st.slider("**ST Depression**", min_value=0.0, max_value=10.0, step=0.1)
        slope = st.selectbox("**Slope of Peak Exercise ST Segment (0-2)**", [0, 1, 2])
        ca = st.selectbox("**Major Vessels Colored (0-3)**", [0, 1, 2, 3])
        thal = st.radio("**Thalassemia Type:**",("**Fixed Defect**","**Normal**","**Reversible Defect**"))
        if thal=="**Fixed Defect**":
            thal=1
        elif thal=="**Normal**":
            thal=2
        else:
            thal=3
    st.markdown("---")
    st.subheader("**Prediction :**")
    if st.button("Predict"):
        user_inputs=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        input_scaled = scaler.transform(np.array(user_inputs).reshape(1, -1))
        pred=model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)[0][1] * 100  # probability of having disease

        if pred[0]== 1:
            st.warning("🫀 The model predicts that this patient **is likely to have heart disease.**")
            st.write("\n")
            st.write(f"**Prediction Confidence :** {proba:.2f}%")
            st.markdown(">**“Higher confidence indicates the model is more certain about the result.**")
            st.markdown("---")
            st.subheader("⚠️ **Possible contributing factors:**")
            st.markdown("""
                        - High cholesterol or blood pressure  
                        - Low maximum heart rate achieved  
                        - Exercise-induced angina or chest pain  
                        """)
            st.info("**💡Consider consulting a cardiologist and maintaining a heart-healthy lifestyle.**")
        else:
            st.success("🫀 The model predicts that this patient **is unlikely to have heart disease.**")
            st.write("\n")
            st.write(f"**Prediction Confidence :** {100 - proba:.2f}%")
            st.markdown(">**“Higher confidence indicates the model is more certain about the result.**")
            st.markdown("---")
            st.success("✅ **Great! Keep your heart healthy.**")
            st.markdown("""
                        - Continue regular exercise 
                        - Maintain a balanced diet   
                        - Monitor blood pressure & cholesterol regularly   
                        """)

elif choice=="📈 Model Performance":
    st.title("📈 Model Performance")
    st.write("""
    >This section provides an overview of how well our trained model performs on test data.
    It helps evaluate the model’s accuracy and reliability before making predictions.
    """)
    st.markdown("---")
    st.subheader("**Performance Metrics**")

    metrics = {
        "Accuracy": 0.90,
        "Precision": 0.93,
        "Recall": 0.87,
        "F1 Score": 0.90
    }
    metrics = {k: f"{v:.2f}" for k, v in metrics.items()}
    st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
    st.markdown(""">The model achieved **90% accuracy** on the test data, showing strong generalization performance.  
                   - **Precision (93%)** indicates it correctly identifies most true positive heart disease cases.  
                   - **Recall (87%)** shows good sensitivity in detecting actual positive cases.  
                   - **F1 Score (90%)** balances precision and recall effectively.""")
    st.markdown("---")
    st.title("**Confusion Matrix**")
    cm = np.array([[13, 1],
          [2, 14]])

    fig, ax = plt.subplots(figsize=(2, 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test Data)")
    st.pyplot(fig,use_container_width=False)
    st.write("\n")
    st.markdown("""
    ###  **Confusion Matrix Analysis**
    > The model correctly predicted most cases — only **1 false positive** and **2 false negatives**, 
    showing strong reliability in detecting heart disease.
    """)
    st.markdown("---")
    st.title("**ROC Curve**")
    st.image(image2, width="content")
    st.markdown("""
    ### ROC Curve Interpretation

    The **ROC (Receiver Operating Characteristic)** curve evaluates the model’s ability to distinguish between patients **with and without heart disease**.

    - The **x-axis** represents the *False Positive Rate (1 - Specificity)* — cases incorrectly predicted as positive.  
    - The **y-axis** represents the *True Positive Rate (Recall)* — cases correctly predicted as positive.  
    - The **blue curve** shows how well the model separates the two classes.  
    - The **red dashed line** represents a random classifier *(AUC = 0.5).*  
    - The **AUC (Area Under Curve) = 0.98**, meaning the model performs **exceptionally well** at distinguishing between patients with and without heart disease.
    """)

else:
    st.title("️ **ℹ️ About**")
    st.markdown("---")
    st.markdown("""
    -This app was created as part of a Machine Learning project to predict **heart disease likelihood** using patient health data.

    -The model analyzes key medical parameters to assist in early detection.  
    
    -Built with **Python, Scikit-learn, and Streamlit** for an interactive user experience.

    ---
    👨‍💻 **Developed by:** Auric Dutt 
    
    """)








