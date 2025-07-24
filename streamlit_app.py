
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go


# Set page config
st.set_page_config(
    page_title="Employee performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for the app
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .h3 { 
        font-size: 1rem;
        color: #1d3557;
        text-align: left;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1d3557;
        margin-bottom: 0.5rem;
    }
    .info-text {
        background-color: #f1faee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #a8dadc;
        color: #1d3557;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        color: #856404;
    }
    .stButton>button {
        background-color: #457b9d;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1d3557;
    }
    /* Improve general text visibility */
    p, h1, h3, label {
        color: #1d3557;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model from disk"""
    try:
        with open('employee_performance_model_.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure employee_performance_model_.pkl is in the current directory.")
        return None
@st.cache_data
def load_feature_info():
    """Return information about features for the prediction form"""
    return {
        'Age': {'type': 'number', 'min': 18, 'max': 100, 'help': 'Employee age in years'},
        'Gender' : {'type' : 'select', 'options': ['Male', 'Female'], 'help': 'Employee Gender'},
        'EducationBackground' : {'type' : 'select', 'options': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'], 'help': 'Employee Educaton background'},
        'MaritalStatus': {'type' : 'select', 'options': ['Single', 'Married', 'Divorced'], 'help': 'Employee Martal status'},
        'EmpDepartment': {'type' : 'select', 'options': ['Sales', 'Development', 'Research & Development', 'Human Resources', 'Finance', 'Data Science'], 'help': 'Select employee department'},
        'EmpJobRole' : {'type' : 'select', 'options': ['Sales Executive', 'Developer', 'Manager', 'Research Scientist', 'Sales Representative', 'Laboratory Technician', 'Senior Developer', 'Technical Lead', 'Business Analyst', 'Finance Manager', 'Senior Manager R&D', 'Healthcare Representative', 'Data Scientist', 'Research Director', 'Delivery Manager'], 'help': 'Select employee role'},
        'BusinessTravelFrequency' : {'type' : 'select', 'options': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 'help' : 'Select Employee travel frequency'},
        'DistanceFromHome' : {'type': 'number', 'min': 1, 'max': 35, 'help': 'Distance from home'},
        'EmpEducationLevel' : {'type' : 'select', 'options': ['Below College', 'College', 'Bachelor', 'Master', 'Doctor'], 'help': 'Select employee Education level'},
        'EmpEnvironmentSatisfaction' : {'type' : 'select', 'options': ['Low', 'Medium', 'High', 'Very High'], 'help': 'Select employee Environment satisfaction'},
        'EmpHourlyRate' :  {'type': 'number', 'min': 30, 'max': 100, 'help': 'Employee Hourly rate'},
        'EmpJobInvolvement' : {'type' : 'select', 'options': ['Low', 'Medium', 'High', 'Very High'], 'help': 'Select employee Job Involvement'},
        'EmpJobLevel' : {'type': 'number', 'min': 1, 'max': 5, 'help': 'Employee Job level'},
        'EmpJobSatisfaction' : {'type' : 'select', 'options': ['Low', 'Medium', 'High', 'Very High'], 'help': 'Select employee Job Involvement'},
        'NumCompaniesWorked' : {'type': 'number', 'min': 0, 'max': 10, 'help': 'Employee Job level'},
        'OverTime' :  {'type' : 'select', 'options': ['Yes', 'No'], 'help': 'Is Employee work Overtime?'},
        'EmpLastSalaryHikePercent' : {'type': 'number', 'min': 10, 'max': 25, 'help': 'Employee Salary Hike %'},
        'EmpRelationshipSatisfaction' : {'type' : 'select', 'options': ['Low', 'Medium', 'High', 'Very High'], 'help': 'Select employee Relationship satisfaction'},
        'TotalWorkExperienceInYears' : {'type': 'number', 'min': 0, 'max': 40, 'help': 'Employee Experience in years'},
        'TrainingTimesLastYear' : {'type': 'number', 'min': 0, 'max': 40, 'help': 'Employee training last year'},
        'EmpWorkLifeBalance' : {'type' : 'select', 'options': ['Bad', 'Good', 'Better', 'Best'], 'help': 'Select employee life style satisfaction'},
        'ExperienceYearsAtThisCompany' : {'type': 'number', 'min': 0, 'max': 40, 'help': 'Employee experience in this company'},
        'ExperienceYearsInCurrentRole' : {'type': 'number', 'min': 0, 'max': 15, 'help': 'Employee experience in Current role'},
        'YearsSinceLastPromotion' : {'type': 'number', 'min': 0, 'max': 15, 'help': 'Employee experience in since last promotion'},
        'YearsWithCurrManager' : {'type': 'number', 'min': 0, 'max': 17, 'help': 'Employee year with current manager'},
        'Attrition' :  {'type' : 'select', 'options': ['Yes', 'No'], 'help': 'Employee attrition'}
     }

def preprocess_input(input_data, model):
    """Process input data to match model's expected format"""
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Create engineered features similar to training

    input_df['Experience_At_Company_Ratio'] = np.where(input_df['TotalWorkExperienceInYears'] > 0,
                                               input_df['ExperienceYearsAtThisCompany'] / input_df['TotalWorkExperienceInYears'],
                                               0)
    
    input_df['Promotion_Lag_Ratio'] = np.where(input_df['ExperienceYearsAtThisCompany'] > 0,
                                       input_df['YearsSinceLastPromotion'] / input_df['ExperienceYearsAtThisCompany'],
                                       0)
    input_df['Age_WorkExperience_Interaction'] = input_df['Age'] * input_df['TotalWorkExperienceInYears']

    job_satisfaction_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
    }

    work_life_balance_mapping = {
        'Bad': 1,
        'Good': 2,
        'Better': 3,
        'Best': 4
    }

    input_df['EmpJobSatisfaction_Numeric'] = input_df['EmpJobSatisfaction'].map(job_satisfaction_mapping)
    input_df['EmpWorkLifeBalance_Numeric'] = input_df['EmpWorkLifeBalance'].map(work_life_balance_mapping)

    input_df['Satisfaction_Balance_Interaction'] = input_df['EmpJobSatisfaction_Numeric'] * input_df['EmpWorkLifeBalance_Numeric']

    # Drop the temporary numeric columns
    input_df = input_df.drop(columns=['EmpJobSatisfaction_Numeric', 'EmpWorkLifeBalance_Numeric'])

    # Add missing columns that the model expects
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            if 'Experience' in col or 'Years' in col:
                input_df[col] = 0
            else:
                input_df[col] = np.nan
    # Ensure the input DataFrame has the same columns as the model
    input_df = input_df.reindex(columns=expected_columns, fill_value=0) 
    # Handle categorical variables
    categorical_cols = input_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in expected_columns:
            input_df[col] = input_df[col].astype('category')
            input_df[col] = input_df[col].cat.codes
    # Fill NaN values with 0
    input_df = input_df.fillna(0)

    return input_df

def get_prediction_and_probability(model, input_df):
    """Get prediction and probability from the model to be displayed in class names [Good, excellent, outstanding]""" 
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    # Map predictions to class names
    class_names = ['Good', 'Excellent', 'Outstanding']
    prediction_class = class_names[prediction[0]]

    # Get the probability for the predicted class
    predicted_prob = probability[0][prediction[0]]

    return prediction_class, predicted_prob
  
def display_prediction_result(result, probability):
    """Display the prediction result in a user-friendly format [Good, Excellent, Outstanding]"""
    st.markdown("<div class='sub-header'>Prediction Result</div>", unsafe_allow_html=True)
    # Display prediction result

    col1, col2 = st.columns([1, 1])

    with col1:

        if result[0] == 'Good':
            st.markdown(f"""
                <div class='warning-box'>
                    <h3> Prediction: {result[0]}</h3>
                    <p>The model predicts that the employee is likely to perform at a {result[0]} level.</p>
                    <p>Confidence: {result[1]:.2%}</p>
                """, unsafe_allow_html=True)
        elif result[0] == 'Excellent':
            st.markdown(f"""
                <div class='success-box'>
                    <h3> Prediction: {result[0]}</h3>
                    <p>The model predicts that the employee is likely to perform at an {result[0]} level.</p>
                    <p>Confidence: {result[1]:.2%}</p>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='success-box'>
                    <h3> Prediction: {result[0]}</h3>
                    <p>The model predicts that the employee is likely to perform at an {result[0]} level.</p>
                    <p>Confidence: {result[1]:.2%}</p>"""
            , unsafe_allow_html=True)
    with col2:
        # Display an image based on the prediction
        if result[0] == 'Good':
            st.markdown("<div class='info-text'>"
                        "The employee is likely to perform at a Good level. Consider providing additional training or support to improve performance.</div>", unsafe_allow_html=True)
        elif result[0] == 'Excellent':
            st.markdown("<div class='info-text'>"
                        "The employee is likely to perform at an Excellent level. Recognize their contributions and consider them for future leadership roles.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='info-text'>"
                        "The employee is likely to perform at an Outstanding level. They are a valuable asset to the team and should be recognized for their exceptional performance.</div>", unsafe_allow_html=True)

def display_feature_importance(model, input_df):
    """Display feature importance for the prediction"""
    st.markdown("<div class='sub-header'>Feature Importance Analysis</div>", unsafe_allow_html=True)

    # Check if model is a pipeline
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        feature_importance = None

        # Process input data through the preprocessing pipeline
        processed_input = model['preprocessor'].transform(input_df)

        if hasattr(model['model'], 'feature_importances_'):
            feature_importance = model['model'].feature_importances_
        elif hasattr(model['model'], 'coef_'):
            feature_importance = model['model'].coef_[0]

        if feature_importance is not None:
            # Try to get feature names
            try:
                feature_names = model['preprocessor'].get_feature_names_out()
            except:
                # If can't get names, create generic ones
                feature_names = [f"Feature {i}" for i in range(len(feature_importance))]

            # Create a DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(10)

            # Plot feature importance
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title='Top 10 Feature Importance',
                        color='Importance', color_continuous_scale='Viridis')

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model does not provide feature importance or coefficients.")

    else:
        st.warning("The loaded model does not support feature importance analysis. Please ensure the model is a tree-based model or has coefficients.")

def generate_sample_data():
    """Generate a sample dataset for demonstration purposes"""
    import random

    # Create a sample dataset with 100 patients
    sample_size = 100
    feature_info = load_feature_info()

    data = {
        'Age': [random.randint(18, 60) for _ in range(sample_size)],
        'Gender': [random.choice(['Male', 'Female']) for _ in range(sample_size)],
        'EducationBackground': [random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources']) for _ in range(sample_size)],
        'MaritalStatus': [random.choice(['Single', 'Married', 'Divorced']) for _ in range(sample_size)],
        'EmpDepartment': [random.choice(['Sales', 'Development', 'Research & Development', 'Human Resources',
                                                'Finance', 'Data Science']) for _ in range(sample_size)],
        'EmpJobRole': [random.choice(['Sales Executive', 'Developer', 'Manager', 'Research Scientist',
                                                'Sales Representative', 'Laboratory Technician', 'Senior Developer',
                                                'Technical Lead', 'Business Analyst', 'Finance Manager',
                                                'Senior Manager R&D', 'Healthcare Representative', 'Data Scientist',
                                                'Research Director', 'Delivery Manager']) for _ in range(sample_size)],
        'BusinessTravelFrequency': [random.choice(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']) for _ in range(sample_size)],   
        'DistanceFromHome': [random.randint(1, 35) for _ in range(sample_size)],
        'EmpEducationLevel': [random.choice(['Below College', 'College', 'Bachelor', 'Master', 'Doctor']) for _ in range(sample_size)],
        'EmpEnvironmentSatisfaction': [random.choice(['Low', 'Medium', 'High', 'Very High']) for _ in range(sample_size)],
        'EmpHourlyRate': [random.randint(30, 100) for _ in range(sample_size)],
        'EmpJobInvolvement': [random.choice(['Low', 'Medium', 'High', 'Very High']) for _ in range(sample_size)],
        'EmpJobLevel': [random.randint(1, 5) for _ in range(sample_size)],
        'EmpJobSatisfaction': [random.choice(['Low', 'Medium', 'High', 'Very High']) for _ in range(sample_size)],
        'NumCompaniesWorked': [random.randint(0, 10) for _ in range(sample_size)],
        'OverTime': [random.choice(['Yes', 'No']) for _ in range(sample_size)],
        'EmpLastSalaryHikePercent': [random.randint(10, 25) for _ in range(sample_size)],
        'EmpRelationshipSatisfaction': [random.choice(['Low', 'Medium', 'High', 'Very High']) for _ in range(sample_size)],
        'TotalWorkExperienceInYears': [random.randint(0, 40) for _ in range(sample_size)],
        'TrainingTimesLastYear': [random.randint(0, 40) for _ in range(sample_size)],
        'EmpWorkLifeBalance': [random.choice(['Bad', 'Good', 'Better', 'Best']) for _ in range(sample_size)],
        'ExperienceYearsAtThisCompany': [random.randint(0, 40) for _ in range(sample_size)],
        'ExperienceYearsInCurrentRole': [random.randint(0, 15) for _ in range(sample_size)],
        'YearsSinceLastPromotion': [random.randint(0, 15) for _ in range(sample_size)],
        'YearsWithCurrManager': [random.randint(0, 17) for _ in range(sample_size)],
        'Attrition': [random.choice(['Yes', 'No']) for _ in range(sample_size)],
        'Performance': [random.choice(['Good', 'Excellent', 'Outstanding']) for _ in range(sample_size)],
    }
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def display_eda_visualizations():
    """Display EDA visualizations"""

    # Load sample data
    sample_data = generate_sample_data()
    # Ensure sample_data columns are unique
    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs(["Distributions", "Correlations", "Performance Analysis","Performance Factors"])
    with eda_tab1:
        dist_feature = st.selectbox(
            "Select feature to visualize:",
            options=['Age', 'Gender', 'EducationBackground', 'MaritalStatus',
                     'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency'])
        
        # Create histogram with distribution
        fig = px.histogram(
            sample_data, x=dist_feature, color='Performance',
            marginal='box', opacity=0.7,
            color_discrete_map={'Outstanding': '#28a745', 'Excellent': '#a5a728', 'Good': '#dc3545'},
            labels={'Performance': 'Performance Level'},
            title=f"Distribution of {dist_feature} by Performance Rate"
        )
        st.plotly_chart(fig, use_container_width=True)

    with eda_tab2:
        # Correlation heatmap
        st.markdown("**Correlation Heatmap**", unsafe_allow_html=True)
        st.markdown("This heatmap shows the correlation between different features in the dataset. Darker colors indicate stronger correlations.", unsafe_allow_html=True)

        # Select numeric features for correlation analysis
        numeric_features = sample_data.select_dtypes(include=[np.number]).columns.tolist()

        # Create heatmap
        fig2 = px.imshow(
            sample_data[numeric_features].corr().__round__(2),
            color_continuous_scale='RdBu_r',
            title="Correlation Heatmap", text_auto=True,
            aspect="auto",
            labels=dict(color='Correlation Coefficient')
        )
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)
    
    with eda_tab3:
        # Explore performance factors

        st.markdown("**Explore performance factors by comparing specific feature with perfamance rate.**", unsafe_allow_html=True)
        performance_feature = st.selectbox(
            "Select feature to compare with performance feature:",
            options=['EmpJobSatisfaction', 'EmpWorkLifeBalance', 'EmpEnvironmentSatisfaction']
        )
        # Create bar plot for performance factors
        fig3 = px.bar(
            sample_data, x=performance_feature, color='Performance',
            color_discrete_map={'Outstanding': '#28a745', 'Excellent': '#a5a728', 'Good': '#dc3545'},
            title=f"Performance Factors based on {performance_feature}",
            labels={'Performance': 'Performance Level'}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with eda_tab4:
        # Performance Analysis
        st.markdown("<div class='sub-header'>Performance Analysis</div>", unsafe_allow_html=True)

        st.markdown("Examine factors that influence Employee to perform outstanding. this analysis helps identify key areas for improvement and development. We have select top 4 categorical features that contributed most in model performance.")

        # Survival by categorical variables
        cat_features = ['EmpEnvironmentSatisfaction', 'EmpDepartment', 'EmpWorkLifeBalance', 'ExperienceYearsInCurrentRole']
        employee_data = []

        for feature in cat_features:
            feature_data = sample_data.groupby(feature)['Performance'].apply(
                lambda x: (x == 'Outstanding').mean()
            ).reset_index()
            feature_data.columns = ['Category', 'Performance Rate']
            feature_data['Feature'] = feature
            employee_data.append(feature_data)

        employee_df = pd.concat(employee_data, ignore_index=True)

        fig = px.bar(
            employee_df, x='Category', y='Performance Rate', color='Feature',
            facet_col='Feature', facet_col_wrap=2,
            title="Performance Rates by Different Factors",
            labels={'Category': '', 'Performance': 'Performance Rate'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the Streamlit app"""
    st.markdown("<h1 class='main-header'>Employee Performance Prediction</h1>", unsafe_allow_html=True)

    # Home page note in paragraph
    st.write("Welcome to the Employee Performance Prediction App!")
    st.markdown("This app uses machine learning to predict employee performance based on various features. This app is designed to help HR professionals and managers make informed decisions about employee performance and potential career development as well as recruitment process", unsafe_allow_html=True)

    # Create tabs for different app sections
    tab1, tab2, tab3, tab4 = st.tabs(["Information", " Predictions ","EDA Visualizations", "Contact"])

    with tab1:
        st.markdown("<div class='sub-header'>Information</div>", unsafe_allow_html=True)
        st.markdown("""
        This app predicts employee performance based on various features such as age, education background, job role, and more.
        It uses a machine learning model trained on historical employee data to provide insights into potential performance levels.
        """)
        st.markdown("""
        **The following Features catured in the training of this model:** 
        - Age
        - Gender
        - Education Background
        - Marital Status
        - Employee Department
        - Employee Job Role
        - Business Travel Frequency
        - Distance From Home
        - Employee Education Level
        - Employee Environment Satisfaction
        - Employee Hourly Rate
        - Employee Job Involvement
        - Employee Job Level
        - Employee Job Satisfaction
        - Number of Companies Worked    
        - Over Time
        - Employee Last Salary Hike Percent
        - Employee Relationship Satisfaction
        - Total Work Experience in Years
        - Training Times Last Year
        - Employee Work Life Balance
        - Experience Years at This Company
        - Experience Years in Current Role
        - Years Since Last Promotion
        - Years With Current Manager
        - Attrition
""")
        
        st.markdown("**Model Performance:**", unsafe_allow_html=True)
        
        st.markdown("""The model has been trained and evaluated on a dataset of employee records, achieving the following metrics on the test set """)
        st.markdown("""
                    - Accuracy: 0.9333
                    - Precision: 0.9334
                    - Recall: 0.9333
                    - F1 Score: 0.9333
                    - ROC AUC: 0.9676 """)
        st.markdown(""" The model uses various features to predict employee performance levels, which are categorized as Good, Excellent, or Outstanding.
        """)
        st.markdown("""
        **How to Use the App:** 
        1. Navigate to the "Predictions" tab to input employee data.
        2. Fill in the required fields based on the employee's information.
        3. Click the "Predict Performance" button to get the prediction result.
        4. View the prediction result and feature importance analysis.
        5. Explore the "EDA Visualizations" tab for insights into the dataset.
        6. For any queries or feedback, visit the "**Contact**" tab.
        """)

    with tab2:
        st.markdown("<div class='sub-header'>Employee Information</div>", unsafe_allow_html=True)
        st.markdown("Please fill the below employee information and Click Predict Performance Button at the end of the form: ", unsafe_allow_html=True)   
        # Load model
        # ...existing code...

        model = load_model()
        if model is None:
            st.stop()

        # ...existing code...

        feature_info = load_feature_info()
        input_data = {}

        for feature, info in feature_info.items():
            if info['type'] == 'number':
                input_data[feature] = st.number_input(
                    label=feature,
                    min_value=info.get('min', 0),
                    max_value=info.get('max', 100),
                    help=info['help']
                )
            elif info['type'] == 'select':
                input_data[feature] = st.selectbox(
                    label=feature,
                    options=info['options'],
                    help=info['help']
                )
            else:
                st.error(f"Unsupported input type for {feature}")
    

        # Prediction button
        if st.button("Predict Performance"):
            with st.spinner('Processing...'):
                input_df = preprocess_input(input_data, model)
                result = get_prediction_and_probability(model, input_df)
                predicted_prob = result[1]
                display_prediction_result(result, predicted_prob)
                display_feature_importance(model, input_df)

    with tab3:
        st.markdown("<div class='sub-header'>Sample data visualization</div>", unsafe_allow_html=True)
        display_eda_visualizations()


    with tab4:
        st.markdown("<div class='sub-header'>Contact Information</div>", unsafe_allow_html=True)
        st.markdown("""
        This app is developed as part of a project to demonstrate employee performance prediction using machine learning.
        It allows users to input employee data and receive predictions on their performance levels.
        """)
        st.markdown("""
                - **Developed by:** Daniel Ndamukunda
                - **Contact:** [LinkedIn](https://www.linkedin.com/in/ndamukunda-daniel/)
                - **Email:** [Gmail](ndamukunda139@gmail.com)
                - **GitHub:** [GitHub Repository](https://github.com/vidacarpeta123/employee_performance_predictions)
                    
    """)

if __name__ == "__main__":  
    main()