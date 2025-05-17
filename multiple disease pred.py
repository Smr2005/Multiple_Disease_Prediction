# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import sys

import importlib

# Check if required packages are installed, if not install them
required_packages = {
    'streamlit_option_menu': 'streamlit-option-menu',
    'scikit-learn': 'scikit-learn',  # Changed from 'sklearn' to 'scikit-learn'
    'pandas': 'pandas'
}

for package_name in required_packages.values():
    try:
        # Try to import the package to check if it's installed
        if package_name == 'scikit-learn':
            # For scikit-learn, we need to check sklearn module
            importlib.import_module('sklearn')
            # Also explicitly check for sklearn.model_selection
            importlib.import_module('sklearn.model_selection')
        elif package_name == 'streamlit-option-menu':
            importlib.import_module('streamlit_option_menu')
        else:
            importlib.import_module(package_name)
    except ImportError:
        st.info(f"Installing required package: {package_name}...")



# Now import after ensuring it's installed
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Add custom CSS for print compatibility with dark mode
st.markdown("""
<style>
    @media print {
        body {
            color: black !important;
            background-color: white !important;
        }
        .stApp {
            color: black !important;
            background-color: white !important;
        }
        .main {
            color: black !important;
            background-color: white !important;
        }
        h1, h2, h3, p {
            color: black !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add a title and description to the main page
st.title("Multiple Disease Prediction System")
st.markdown("""
This application uses machine learning to predict the likelihood of various diseases based on patient data.
Please select a disease from the sidebar to begin.

**Note:** This tool is for educational purposes only and should not replace professional medical advice.

### About This System
This system uses advanced machine learning algorithms to analyze patient data and predict the likelihood of various diseases including:

- **Diabetes**: Analyzes factors like glucose levels, BMI, and family history
- **Heart Disease**: Evaluates cardiovascular health indicators
- **Parkinson's Disease**: Examines voice and movement-related measurements
- **Lung Cancer**: Assesses respiratory symptoms and risk factors
- **Hypothyroid**: Analyzes thyroid function indicators

Each prediction model has been trained on medical datasets and provides risk assessments based on the input parameters.
""")

# Loading the saved models
# We need to train models first since we only have CSV data files
import pandas as pd

# Import sklearn modules with proper error handling
try:
    # First try to import the base sklearn module
    import sklearn
    # Then import specific modules
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Verify sklearn is properly installed by using it
    # Get version safely
    try:
        sklearn_version = sklearn.__version__
        # Store version info in session state instead of displaying directly
        st.session_state['sklearn_version'] = sklearn_version
    except AttributeError:
        # If __version__ is not available, just confirm it's imported
        st.session_state['sklearn_version'] = "Unknown"

except ImportError as e:
    st.error(f"Error importing sklearn modules: {e}")
    st.info("Attempting to reinstall scikit-learn...")

    # After reinstalling, try to import again
    try:
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        # Try to get version again
        try:
            sklearn_version = sklearn.__version__
            st.session_state['sklearn_version'] = sklearn_version
        except AttributeError:
            st.session_state['sklearn_version'] = "Unknown"

    except ImportError as e:
        st.error(f"Still having issues with scikit-learn: {e}")
        st.error("Please try restarting the application or installing scikit-learn manually.")
        sys.exit(1)  # Exit if sklearn cannot be imported after reinstall attempt


# Create a directory for saved models if it doesn't exist
import os
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Function to load a model or train it if the pickle file doesn't exist
def load_or_train_model(model_name, data_path, target_column, drop_columns=None):
    pickle_path = f'saved_models/{model_name}.pkl'

    # Try to load the model from pickle file
    try:
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
        # Store model loading info in session state instead of displaying directly
        if 'loaded_models' not in st.session_state:
            st.session_state['loaded_models'] = []
        st.session_state['loaded_models'].append(model_name)
        return model, True
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        st.warning(f"Saved model for {model_name} not found or corrupted. Training a new model...")

        try:
            # Load the data
            data = pd.read_csv(data_path)

            # Drop non-feature columns if specified
            if drop_columns:
                data = data.drop(drop_columns, axis=1)

            # Extract features and target
            if target_column in data.columns:
                X = data.drop(target_column, axis=1)
                y = data[target_column]
            else:
                # For heart disease data where target might be the last column without a name
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            # Train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Save the model for future use
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(model, f)
                # Store training info in session state instead of displaying directly
                if 'trained_models' not in st.session_state:
                    st.session_state['trained_models'] = []
                st.session_state['trained_models'].append(model_name)
            except Exception as e:
                st.warning(f"Could not save {model_name} model: {e}")

            return model, True
        except FileNotFoundError:
            st.error(f"Dataset for {model_name} not found. This prediction will be disabled.")
            return None, False
        except Exception as e:
            st.error(f"Error training {model_name} model: {e}")
            return None, False

# Load or train diabetes model
diabetes_model, _ = load_or_train_model(
    model_name='diabetes_model',
    data_path='diabetes.csv',
    target_column='Outcome'
)

# Load diabetes data for feature names
try:
    diabetes_data = pd.read_csv('diabetes.csv')
    X_diabetes = diabetes_data.drop('Outcome', axis=1)
except:
    st.error("Could not load diabetes dataset for feature names.")

# Load or train heart disease model
heart_disease_model, _ = load_or_train_model(
    model_name='heart_disease_model',
    data_path='heart.csv',
    target_column='target'  # This might be different in your dataset
)

# Load heart data for feature names
try:
    heart_data = pd.read_csv('heart.csv')
    X_heart = heart_data.iloc[:, :-1]
except:
    st.error("Could not load heart disease dataset for feature names.")

# Load or train Parkinson's model
parkinsons_model, _ = load_or_train_model(
    model_name='parkinsons_model',
    data_path='parkinsons.csv',
    target_column='status',
    drop_columns=['name']
)

# Load Parkinson's data for feature names
try:
    parkinsons_data = pd.read_csv('parkinsons.csv')
    parkinsons_data = parkinsons_data.drop('name', axis=1)
    X_parkinsons = parkinsons_data.drop('status', axis=1)
except:
    st.error("Could not load Parkinson's dataset for feature names.")

# Load or train lung cancer model
lung_cancer_model, lung_cancer_model_ready = load_or_train_model(
    model_name='lung_cancer_model',
    data_path='lung_cancer.csv',
    target_column='Cancer'
)

# Load lung cancer data for feature names if model is ready
if lung_cancer_model_ready:
    try:
        lung_cancer_data = pd.read_csv('lung_cancer.csv')
        X_lung_cancer = lung_cancer_data.drop('Cancer', axis=1)
    except:
        st.error("Could not load lung cancer dataset for feature names.")

# Load or train hypothyroid model
hypothyroid_model, hypothyroid_model_ready = load_or_train_model(
    model_name='hypothyroid_model',
    data_path='hypothyroid.csv',
    target_column='Hypothyroid'
)

# Load hypothyroid data for feature names if model is ready
if hypothyroid_model_ready:
    try:
        hypothyroid_data = pd.read_csv('hypothyroid.csv')
        X_hypothyroid = hypothyroid_data.drop('Hypothyroid', axis=1)
    except:
        st.error("Could not load hypothyroid dataset for feature names.")


#Sidebar for navigators
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Lung Cancer Prediction',
                            'Hypothyroid Prediction'],
                           icons = ['activity', 'heart', 'person', 'lungs', 'thermometer'],
                           default_index = 0)
    
    # Add technical information in a collapsible section at the bottom of sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("Technical Information", expanded=False):
        st.write(f"**scikit-learn version:** {st.session_state.get('sklearn_version', 'Unknown')}")
        
        # Display loaded models
        if 'loaded_models' in st.session_state and st.session_state['loaded_models']:
            st.write("**Loaded models:**")
            for model in st.session_state['loaded_models']:
                st.write(f"- {model}")
        
        # Display trained models
        if 'trained_models' in st.session_state and st.session_state['trained_models']:
            st.write("**Newly trained models:**")
            for model in st.session_state['trained_models']:
                st.write(f"- {model}")
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):

    #page title
    st.title('Diabetes Prediction Using ML')

    st.write("""
    ### About Diabetes
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). 
    It occurs when your body either doesn't produce enough insulin or can't effectively use the insulin it produces.
    
    **Types of Diabetes:**
    - **Type 1 Diabetes**: The body doesn't produce insulin. Usually diagnosed in children and young adults.
    - **Type 2 Diabetes**: The body doesn't use insulin properly. Most common form of diabetes.
    - **Gestational Diabetes**: Develops during pregnancy in women who don't already have diabetes.
    
    **Common Symptoms:**
    - Frequent urination
    - Increased thirst and hunger
    - Unexplained weight loss
    - Fatigue
    - Blurred vision
    - Slow-healing sores
    
    **Risk Factors:**
    - Family history
    - Obesity
    - Physical inactivity
    - Age (risk increases with age)
    - High blood pressure
    - Abnormal cholesterol levels
    
    This tool uses several health metrics to assess the likelihood of diabetes based on machine learning analysis of patient data.
    
    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    # Example values for diabetes
    diabetes_example = {
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    }

    # Add example values button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values (High Risk)', key='diabetes_example'):
            st.info("These values represent a high-risk case that is likely to show a positive prediction.")
            for key, value in diabetes_example.items():
                st.session_state[key] = value

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
       Pregnancies = st.text_input('Number of Pregnancies',
                                  key='Pregnancies',
                                  help='Number of times pregnant (0 for males)')

    with col2:
       Glucose = st.text_input('Glucose Level (mg/dL)',
                              key='Glucose',
                              help='Plasma glucose concentration after 2 hours in an oral glucose tolerance test. Normal fasting: 70-99 mg/dL')

    with col3:
       BloodPressure = st.text_input('Blood Pressure (mm Hg)',
                                    key='BloodPressure',
                                    help='Diastolic blood pressure. Normal: <80 mm Hg')

    with col1:
       SkinThickness = st.text_input('Skin Thickness (mm)',
                                    key='SkinThickness',
                                    help='Triceps skin fold thickness. Measures fat content')

    with col2:
       Insulin = st.text_input('Insulin Level (μU/ml)',
                              key='Insulin',
                              help='2-Hour serum insulin. Normal fasting: <25 μU/ml')

    with col3:
       BMI = st.text_input('BMI value',
                          key='BMI',
                          help='Body Mass Index. Normal: 18.5-24.9')

    with col1:
       DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function',
                                               key='DiabetesPedigreeFunction',
                                               help='Scores likelihood of diabetes based on family history')

    with col2:
       Age = st.text_input('Age (years)',
                          key='Age',
                          help='Age in years')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):
        try:
            # Create a dictionary with feature names matching the training data
            input_dict = {
                'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age
            }

            # Convert to DataFrame with proper feature names
            input_df = pd.DataFrame([input_dict])

            # Convert string values to float
            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            # Make prediction using DataFrame with feature names
            diab_prediction = diabetes_model.predict(input_df)
            
            # Get probability scores (percentage chance of diabetes)
            diab_probability = diabetes_model.predict_proba(input_df)
            # Check if the model has predicted probabilities for both classes
            if diab_probability.shape[1] > 1:
                diabetes_percentage = round(diab_probability[0][1] * 100, 2)  # Probability of class 1 (diabetes)
            else:
                # If only one class was in the training data, use the prediction directly
                diabetes_percentage = round(diab_probability[0][0] * 100, 2) if diab_prediction[0] == 1 else 0.0
            
            if diab_prediction[0] == 1:
                diab_diagnosis = f'The person is likely to have diabetes (Risk: {diabetes_percentage}%)'
                
                # Personalized recommendations based on input values
                recommendations = []
                try:
                    glucose_value = float(Glucose)
                    bmi_value = float(BMI)
                    age_value = float(Age)
                    
                    if glucose_value > 140:
                        recommendations.append("• Your glucose level is significantly elevated. Consider monitoring your blood sugar regularly.")
                    
                    if bmi_value > 30:
                        recommendations.append("• Your BMI indicates obesity, which increases diabetes risk. A weight management plan may help.")
                    
                    if age_value > 45:
                        recommendations.append("• Age is a risk factor. Regular diabetes screenings are important.")
                    
                    if float(DiabetesPedigreeFunction) > 0.5:
                        recommendations.append("• Your family history score indicates increased genetic risk.")
                except:
                    pass
                
                if not recommendations:
                    recommendations.append("• Maintain a healthy diet and regular exercise routine.")
                
                st.warning(f"""
                **What is Diabetes?**
                Diabetes is a metabolic disease that causes high blood sugar levels. The hormone insulin
                moves sugar from the blood into your cells to be stored or used for energy. With diabetes,
                your body either doesn't make enough insulin or can't effectively use the insulin it makes.

                **Your Risk Assessment: {diabetes_percentage}% risk of diabetes**
                
                **Personalized Recommendations:**
                {chr(10).join(recommendations)}
                
                **Next Steps:**
                With a {diabetes_percentage}% risk assessment, we strongly recommend consulting with a healthcare 
                provider for proper testing and diagnosis. This prediction is not a medical diagnosis but 
                can help guide your health decisions.
                """)
            else:
                diab_diagnosis = f'The person is not likely to have diabetes (Risk: {diabetes_percentage}%)'
                st.success(diab_diagnosis)
                
                # Provide health advice even for negative predictions
                recommendations = []
                try:
                    glucose_value = float(Glucose)
                    bmi_value = float(BMI)
                    
                    if glucose_value > 99:
                        recommendations.append("• Your glucose level appears to be elevated, which can be a risk factor for prediabetes.")
                    
                    if bmi_value > 25:
                        recommendations.append("• Your BMI indicates you may be overweight, which is a risk factor for diabetes.")
                except:
                    pass
                
                st.info(f"""
                **Your Risk Assessment: {diabetes_percentage}% risk of diabetes**
                
                **Healthy Habits to Maintain:**
                • Continue regular exercise (at least 150 minutes per week)
                • Maintain a balanced diet rich in vegetables, fruits, and whole grains
                • Limit processed foods and added sugars
                • Get regular health check-ups, especially if you have risk factors
                
                {chr(10).join(recommendations) if recommendations else ""}
                
                Even with a low risk assessment, maintaining healthy habits is important for preventing diabetes.
                """)

        except ValueError:
            st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
        except Exception as e:
            st.error(f"An error occurred: {e}")



# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    #page title
    st.title('Heart Disease Prediction Using ML')

    st.write("""
    ### About Heart Disease
    Heart disease refers to several types of heart conditions. The most common is coronary artery disease (CAD),
    which can lead to heart attack. Heart disease is the leading cause of death for both men and women worldwide.
    
    **Types of Heart Disease:**
    - **Coronary Artery Disease**: Narrowing or blockage of coronary arteries
    - **Heart Failure**: Heart cannot pump blood effectively
    - **Arrhythmias**: Abnormal heart rhythms
    - **Valve Disease**: Problems with heart valves
    - **Congenital Heart Defects**: Heart problems present at birth
    
    **Common Symptoms:**
    - Chest pain or discomfort (angina)
    - Shortness of breath
    - Pain or numbness in arms or shoulders
    - Fatigue
    - Irregular heartbeat
    - Dizziness or lightheadedness
    
    **Risk Factors:**
    - High blood pressure
    - High cholesterol
    - Smoking
    - Diabetes
    - Family history
    - Age (risk increases with age)
    - Physical inactivity
    - Obesity
    
    This tool uses various health metrics to assess the likelihood of heart disease based on machine learning analysis of patient data.
    
    **Note:** This tool is for educational purposes only and should not replace professional medical advice.
    """)

    # Example values for heart disease
    heart_example = {
        'age': '63',
        'sex': '1',  # Male
        'cp': '3',   # Chest pain type (3 = asymptomatic)
        'trestbps': '145',
        'chol': '233',
        'fbs': '1',
        'restecg': '0',
        'thalach': '150',
        'exang': '0',
        'oldpeak': '2.3',
        'slope': '0',
        'ca': '0',
        'thal': '1'
    }

    # Add example values button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values (High Risk)', key='heart_example'):
            st.info("These values represent a high-risk case that is likely to show a positive prediction.")
            for key, value in heart_example.items():
                st.session_state[key] = value

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Basic Information", "Clinical Measurements"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age (years)',
                              key='age',
                              help='Age in years')

        with col2:
            sex = st.text_input('Sex (0=Female, 1=Male)',
                              key='sex',
                              help='0 = Female, 1 = Male')

        with col3:
            cp = st.text_input('Chest Pain Type (0-3)',
                             key='cp',
                             help='0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure (mm Hg)',
                                    key='trestbps',
                                    help='Resting blood pressure. Normal: <120 mm Hg')

        with col2:
            chol = st.text_input('Serum Cholesterol (mg/dL)',
                               key='chol',
                               help='Serum cholesterol. Desirable: <200 mg/dL')

        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dL (1=Yes, 0=No)',
                              key='fbs',
                              help='1 = Fasting blood sugar > 120 mg/dL, 0 = Fasting blood sugar <= 120 mg/dL')

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            restecg = st.text_input('Resting ECG Results (0-2)',
                                   key='restecg',
                                   help='0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy')

        with col2:
            thalach = st.text_input('Maximum Heart Rate Achieved',
                                   key='thalach',
                                   help='Maximum heart rate achieved during exercise. Normal max: 220 - age')

        with col3:
            exang = st.text_input('Exercise Induced Angina (1=Yes, 0=No)',
                                key='exang',
                                help='1 = Yes, 0 = No')

        with col1:
            oldpeak = st.text_input('ST Depression Induced by Exercise',
                                   key='oldpeak',
                                   help='ST depression induced by exercise relative to rest')

        with col2:
            slope = st.text_input('Slope of Peak Exercise ST Segment (0-2)',
                                key='slope',
                                help='0 = Upsloping, 1 = Flat, 2 = Downsloping')

        with col3:
            ca = st.text_input('Number of Major Vessels (0-3)',
                             key='ca',
                             help='Number of major vessels colored by fluoroscopy (0-3)')

        with col1:
            thal = st.text_input('Thalassemia (0-3)',
                               key='thal',
                               help='0 = Normal, 1 = Fixed defect, 2 = Reversible defect, 3 = Unknown')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            # Get the column names from the original heart dataset
            feature_names = X_heart.columns.tolist()

            # Create a dictionary with feature names matching the training data
            input_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

            # Convert to DataFrame with proper feature names
            input_df = pd.DataFrame([input_dict])

            # Convert string values to float
            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            # Make prediction using DataFrame with feature names
            heart_prediction = heart_disease_model.predict(input_df)
            
            # Get probability scores (percentage chance of heart disease)
            heart_probability = heart_disease_model.predict_proba(input_df)
            # Check if the model has predicted probabilities for both classes
            if heart_probability.shape[1] > 1:
                heart_percentage = round(heart_probability[0][1] * 100, 2)  # Probability of class 1 (heart disease)
            else:
                # If only one class was in the training data, use the prediction directly
                heart_percentage = round(heart_probability[0][0] * 100, 2) if heart_prediction[0] == 1 else 0.0
            
            if heart_prediction[0] == 1:
                heart_diagnosis = f'The person is likely to have heart disease (Risk: {heart_percentage}%)'
                
                # Personalized recommendations based on input values
                recommendations = []
                try:
                    chol_value = float(chol)
                    bp_value = float(trestbps)
                    age_value = float(age)
                    
                    if chol_value > 240:
                        recommendations.append("• Your cholesterol is high. Consider dietary changes and medication if prescribed.")
                    elif chol_value > 200:
                        recommendations.append("• Your cholesterol is borderline high. Consider dietary improvements.")
                    
                    if bp_value >= 140:
                        recommendations.append("• Your blood pressure indicates hypertension. Follow your doctor's recommendations for management.")
                    elif bp_value >= 120:
                        recommendations.append("• Your blood pressure is elevated. Monitor regularly and consider lifestyle changes.")
                    
                    if float(exang) == 1:
                        recommendations.append("• You have exercise-induced angina, which is a significant indicator of coronary artery disease.")
                    
                    if float(oldpeak) > 2:
                        recommendations.append("• Your ST depression during exercise is concerning and suggests reduced blood flow to the heart.")
                except:
                    pass
                
                if not recommendations:
                    recommendations.append("• Follow a heart-healthy diet and exercise program as recommended by your doctor.")
                
                st.warning(f"""
                **What is Heart Disease?**
                Heart disease describes a range of conditions that affect your heart, including coronary artery disease,
                heart rhythm problems (arrhythmias), and heart defects you're born with (congenital heart defects).

                **Your Risk Assessment: {heart_percentage}% risk of heart disease**
                
                **Personalized Recommendations:**
                {chr(10).join(recommendations)}
                
                **Next Steps:**
                With a {heart_percentage}% risk assessment, we strongly recommend consulting with a cardiologist 
                for proper evaluation and diagnosis. This prediction is not a medical diagnosis but 
                can help guide your health decisions.
                """)
            else:
                heart_diagnosis = f'The person is not likely to have heart disease (Risk: {heart_percentage}%)'
                st.success(heart_diagnosis)
                
                # Provide health advice even for negative predictions
                recommendations = []
                try:
                    chol_value = float(chol)
                    bp_value = float(trestbps)
                    
                    if chol_value > 200:
                        recommendations.append("• Your cholesterol level is above the desirable range, which can be a risk factor for heart disease.")
                    
                    if bp_value > 120:
                        recommendations.append("• Your blood pressure is elevated, which is a risk factor for heart disease.")
                except:
                    pass
                
                st.info(f"""
                **Your Risk Assessment: {heart_percentage}% risk of heart disease**
                
                **Heart-Healthy Habits to Maintain:**
                • Maintain a heart-healthy diet low in saturated fats and sodium
                • Exercise regularly (at least 150 minutes of moderate activity per week)
                • Avoid smoking and limit alcohol consumption
                • Manage stress through relaxation techniques
                • Get regular check-ups and monitor your blood pressure and cholesterol
                
                {chr(10).join(recommendations) if recommendations else ""}
                
                Even with a low risk assessment, maintaining heart-healthy habits is important for prevention.
                """)

        except ValueError:
            st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
        except Exception as e:
            st.error(f"An error occurred: {e}")



# Parkinson's Prediction Page
if (selected == 'Parkinsons Prediction'):

    # page title
    st.title('Parkinsons Prediction Using ML')

    st.write("""
    ### About Parkinson's Disease
    Parkinson's disease is a progressive neurological disorder that affects movement. It develops when neurons in the brain 
    that control movement stop working properly or die, reducing the production of dopamine, a chemical that helps coordinate movement.
    
    **Key Characteristics:**
    - **Progressive**: Symptoms gradually worsen over time
    - **Neurodegenerative**: Involves the degeneration of neurons in the brain
    - **Movement Disorder**: Primarily affects motor function
    
    **Common Symptoms:**
    - Tremor (shaking) at rest
    - Bradykinesia (slowness of movement)
    - Rigidity (stiffness) in limbs
    - Postural instability (balance problems)
    - Changes in speech and voice
    - Micrographia (small, cramped handwriting)
    
    **Voice Changes in Parkinson's:**
    Voice and speech changes are common early indicators of Parkinson's disease. These include:
    - Reduced volume (hypophonia)
    - Monotone speech (lack of inflection)
    - Slurred or mumbled speech
    - Changes in voice quality (breathiness, hoarseness)
    - Abnormal speech rhythm
    
    **About This Analysis Tool:**
    This tool analyzes voice recordings to detect patterns associated with Parkinson's disease.
    The measurements below come from sustained phonations (saying 'ahhh'), where various
    properties of the voice are analyzed using acoustic parameters.
    
    **Note:** If you don't have these measurements, you can use the example values provided
    or consult with a healthcare professional for a proper assessment. This tool is for educational
    purposes only and should not replace professional medical evaluation.
    """)

    # Example values for Parkinson's
    parkinsons_example = {
        'fo': '119.992', 'fhi': '157.302', 'flo': '74.997',
        'Jitter_percent': '0.00662', 'Jitter_Abs': '0.00004',
        'RAP': '0.00401', 'PPQ': '0.00506', 'DDP': '0.01204',
        'Shimmer': '0.04374', 'Shimmer_dB': '0.426',
        'APQ3': '0.02182', 'APQ5': '0.02971', 'APQ': '0.02971',
        'DDA': '0.06545', 'NHR': '0.02211', 'HNR': '21.033',
        'RPDE': '0.414783', 'DFA': '0.815285',
        'spread1': '-4.813031', 'spread2': '0.266482',
        'D2': '2.301442', 'PPE': '0.284654'
    }

    # Add example values button at the top
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('Fill Example Values (High Risk)', key='parkinsons_example'):
            st.info("These values represent a high-risk case that is likely to show a positive prediction.")
            for key, value in parkinsons_example.items():
                st.session_state[key] = value

    # Create tabs for organized input
    tab1, tab2, tab3 = st.tabs(["Frequency Measures", "Amplitude Measures", "Nonlinear Measures"])

    with tab1:
        st.subheader("Frequency Measurements")
        st.write("These measure the fundamental frequency of the voice and its variations.")

        col1, col2 = st.columns(2)

        with col1:
            fo = st.text_input('Average vocal frequency (Hz)',
                              key='fo',
                              help='Normal range: 107-177 Hz for men, 164-258 Hz for women')

            fhi = st.text_input('Maximum vocal frequency (Hz)',
                               key='fhi',
                               help='Usually higher than the average frequency')

            flo = st.text_input('Minimum vocal frequency (Hz)',
                               key='flo',
                               help='Usually lower than the average frequency')

        with col2:
            Jitter_percent = st.text_input('Jitter in percentage (%)',
                                         key='Jitter_percent',
                                         help='Frequency variation - normal range: <1.04%')

            Jitter_Abs = st.text_input('Absolute jitter in microseconds',
                                     key='Jitter_Abs',
                                     help='Frequency variation - normal range: <83.2 μs')

            RAP = st.text_input('Relative Amplitude Perturbation',
                              key='RAP',
                              help='Frequency variation - normal range: <0.680%')

            PPQ = st.text_input('Five-point Period Perturbation',
                              key='PPQ',
                              help='Frequency variation - normal range: <0.840%')

            DDP = st.text_input('Average difference of differences',
                              key='DDP',
                              help='Frequency variation - normal range: <2.040%')

    with tab2:
        st.subheader("Amplitude Measurements")
        st.write("These measure the amplitude variations in the voice.")

        col1, col2 = st.columns(2)

        with col1:
            Shimmer = st.text_input('Shimmer in percentage (%)',
                                  key='Shimmer',
                                  help='Amplitude variation - normal range: <3.810%')

            Shimmer_dB = st.text_input('Shimmer in decibels (dB)',
                                     key='Shimmer_dB',
                                     help='Amplitude variation - normal range: <0.350 dB')

            APQ3 = st.text_input('Three-point Amplitude Perturbation',
                               key='APQ3',
                               help='Amplitude variation - normal range: <3.070%')

        with col2:
            APQ5 = st.text_input('Five-point Amplitude Perturbation',
                               key='APQ5',
                               help='Amplitude variation - normal range: <3.420%')

            APQ = st.text_input('11-point Amplitude Perturbation',
                              key='APQ',
                              help='Amplitude variation - normal range: <3.960%')

            DDA = st.text_input('Average absolute differences',
                              key='DDA',
                              help='Amplitude variation - normal range: <3.590%')

    with tab3:
        st.subheader("Nonlinear Measures & Noise Ratios")
        st.write("These measure noise, nonlinear dynamics, and signal complexity.")

        col1, col2 = st.columns(2)

        with col1:
            NHR = st.text_input('Noise-to-Harmonics Ratio',
                              key='NHR',
                              help='Ratio of noise to tonal components - normal range: <0.190')

            HNR = st.text_input('Harmonics-to-Noise Ratio (dB)',
                              key='HNR',
                              help='Ratio of harmonics to noise - higher values are better')

            RPDE = st.text_input('Recurrence Period Density Entropy',
                               key='RPDE',
                               help='Measures voice irregularity - range: 0 to 1')

            DFA = st.text_input('Detrended Fluctuation Analysis',
                              key='DFA',
                              help='Signal fractal scaling exponent - range: 0.5 to 1')

        with col2:
            spread1 = st.text_input('Frequency variation measure 1',
                                  key='spread1',
                                  help='Nonlinear measure of frequency variation')

            spread2 = st.text_input('Frequency variation measure 2',
                                  key='spread2',
                                  help='Nonlinear measure of frequency variation')

            D2 = st.text_input('Correlation dimension',
                             key='D2',
                             help='Measures complexity of the voice signal')

            PPE = st.text_input('Pitch Period Entropy',
                              key='PPE',
                              help='Measures impaired control of stable pitch - range: 0 to 1')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        try:
            # Get the column names from the original parkinsons dataset
            feature_names = X_parkinsons.columns.tolist()

            # Create a list of input values
            input_values = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            # Create a dictionary with feature names matching the training data
            input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

            # Convert to DataFrame with proper feature names
            input_df = pd.DataFrame([input_dict])

            # Convert string values to float
            for column in input_df.columns:
                input_df[column] = input_df[column].astype(float)

            # Make prediction using DataFrame with feature names
            parkinsons_prediction = parkinsons_model.predict(input_df)
            
            # Get probability scores (percentage chance of Parkinson's)
            parkinsons_probability = parkinsons_model.predict_proba(input_df)
            # Check if the model has predicted probabilities for both classes
            if parkinsons_probability.shape[1] > 1:
                parkinsons_percentage = round(parkinsons_probability[0][1] * 100, 2)  # Probability of class 1 (Parkinson's)
            else:
                # If only one class was in the training data, use the prediction directly
                parkinsons_percentage = round(parkinsons_probability[0][0] * 100, 2) if parkinsons_prediction[0] == 1 else 0.0
            
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = f"The person may have Parkinson's disease (Risk: {parkinsons_percentage}%)"
                
                # Personalized recommendations based on voice metrics
                recommendations = []
                try:
                    # Check for severe voice irregularities
                    if float(Jitter_percent) > 1.0:
                        recommendations.append("• Your voice frequency variation (jitter) is elevated, which is common in Parkinson's.")
                    
                    if float(Shimmer) > 0.1:
                        recommendations.append("• Your voice amplitude variation (shimmer) is elevated, which is common in Parkinson's.")
                    
                    if float(HNR) < 15:
                        recommendations.append("• Your harmonics-to-noise ratio is low, indicating increased breathiness in your voice.")
                    
                    if float(PPE) > 0.2:
                        recommendations.append("• Your pitch period entropy suggests reduced control over vocal stability.")
                except:
                    pass
                
                if not recommendations:
                    recommendations.append("• Consider a comprehensive neurological evaluation to assess all symptoms.")
                
                st.warning(f"""
                **What is Parkinson's Disease?**
                Parkinson's disease is a progressive nervous system disorder that affects movement. Symptoms start 
                gradually, sometimes with a barely noticeable tremor in just one hand. Tremors are common, but the 
                disorder also commonly causes stiffness or slowing of movement.

                **Your Risk Assessment: {parkinsons_percentage}% risk based on voice analysis**
                
                **Personalized Insights:**
                {chr(10).join(recommendations)}
                
                **Next Steps:**
                With a {parkinsons_percentage}% risk assessment based on voice patterns, we recommend consulting with a 
                neurologist for a comprehensive evaluation. This prediction is based solely on voice analysis and 
                should not be considered a medical diagnosis.
                """)
            else:
                parkinsons_diagnosis = f"The person does not show voice patterns associated with Parkinson's disease (Risk: {parkinsons_percentage}%)"
                st.success(parkinsons_diagnosis)
                
                st.info(f"""
                **Your Risk Assessment: {parkinsons_percentage}% risk based on voice analysis**
                
                **Understanding Voice Health:**
                • Voice changes can occur due to many factors including aging, vocal strain, or other conditions
                • Regular vocal exercises can help maintain voice quality
                • If you notice persistent voice changes, consider consulting with a speech pathologist
                
                Even with a low risk assessment, it's important to monitor any changes in movement, coordination, 
                or voice quality over time.
                """)

        except ValueError:
            st.error("Please enter valid numerical values for all fields or use the 'Fill with Example Values' button")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Lung Cancer Prediction Page
if (selected == 'Lung Cancer Prediction'):

    #page title
    st.title('Lung Cancer Prediction Using ML')

    if not lung_cancer_model_ready:
        st.error("Lung cancer prediction is currently unavailable. Please add a lung_cancer.csv file to enable this feature.")
        st.info("The lung cancer dataset should contain relevant features such as age, smoking history, exposure to pollutants, family history, etc., with a target column named 'Cancer'.")
    else:
        st.write("""
        ### About Lung Cancer
        Lung cancer is a type of cancer that begins in the lungs and is the leading cause of cancer deaths worldwide.
        It occurs when cells in the lungs grow uncontrollably, forming tumors that interfere with lung function.
        
        **Types of Lung Cancer:**
        - **Non-Small Cell Lung Cancer (NSCLC)**: Most common type (85% of cases)
          - Adenocarcinoma
          - Squamous Cell Carcinoma
          - Large Cell Carcinoma
        - **Small Cell Lung Cancer (SCLC)**: Less common but more aggressive
        
        **Common Symptoms:**
        - Persistent cough
        - Coughing up blood
        - Chest pain that worsens with deep breathing
        - Hoarseness
        - Shortness of breath
        - Wheezing
        - Unexplained weight loss
        - Fatigue
        
        **Major Risk Factors:**
        - Smoking (responsible for 80-90% of lung cancer cases)
        - Secondhand smoke exposure
        - Radon gas exposure
        - Asbestos exposure
        - Family history
        - Air pollution
        - Previous radiation therapy
        
        This tool uses various risk factors and symptoms to assess the likelihood of lung cancer based on machine learning analysis of patient data.
        
        **Note:** This tool is for educational purposes only and should not replace professional medical advice. Early detection is crucial for successful treatment.
        """)

        # Example values for lung cancer (High risk case)
        lung_cancer_example = {
            'lung_age': '68',            # Older age increases risk
            'gender': '1',               # Male (higher risk)
            'smoking': '1',              # Major risk factor
            'yellow_fingers': '1',       # Sign of heavy smoking
            'anxiety': '1',              # Can be associated with serious illness
            'peer_pressure': '0',        # No
            'chronic_disease': '1',      # Existing lung conditions increase risk
            'fatigue': '1',              # Common symptom
            'allergy': '0',              # No
            'wheezing': '1',             # Key symptom
            'alcohol': '1',              # Can increase risk
            'coughing': '1',             # Key symptom
            'shortness_of_breath': '1',  # Key symptom
            'swallowing_difficulty': '1',# Can indicate advanced disease
            'chest_pain': '1'            # Key symptom
        }

        # Add example values button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button('Fill Example Values (High Risk)', key='lung_cancer_example'):
                st.info("These values represent a high-risk case that is likely to show a positive prediction.")
                for key, value in lung_cancer_example.items():
                    st.session_state[key] = value

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Demographics & Habits", "Symptoms"])

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.text_input('Age (years)',
                                  key='lung_age',
                                  help='Age in years')

            with col2:
                gender = st.text_input('Gender (0=Female, 1=Male)',
                                     key='gender',
                                     help='0 = Female, 1 = Male')

            with col3:
                smoking = st.text_input('Smoking (0=No, 1=Yes)',
                                      key='smoking',
                                      help='Do you smoke or have a history of smoking?')

            with col1:
                yellow_fingers = st.text_input('Yellow Fingers (0=No, 1=Yes)',
                                             key='yellow_fingers',
                                             help='Discoloration of fingers due to smoking')

            with col2:
                alcohol = st.text_input('Alcohol Consumption (0=No, 1=Yes)',
                                      key='alcohol',
                                      help='Regular alcohol consumption')

            with col3:
                peer_pressure = st.text_input('Peer Pressure (0=No, 1=Yes)',
                                            key='peer_pressure',
                                            help='Influenced by peers to smoke or drink')

        with tab2:
            col1, col2, col3 = st.columns(3)

            with col1:
                chronic_disease = st.text_input('Chronic Disease (0=No, 1=Yes)',
                                              key='chronic_disease',
                                              help='Presence of any chronic disease')

            with col2:
                fatigue = st.text_input('Fatigue (0=No, 1=Yes)',
                                      key='fatigue',
                                      help='Persistent fatigue or weakness')

            with col3:
                allergy = st.text_input('Allergy (0=No, 1=Yes)',
                                      key='allergy',
                                      help='Presence of allergies')

            with col1:
                wheezing = st.text_input('Wheezing (0=No, 1=Yes)',
                                       key='wheezing',
                                       help='Noisy breathing or whistling sound when breathing')

            with col2:
                coughing = st.text_input('Coughing (0=No, 1=Yes)',
                                       key='coughing',
                                       help='Persistent coughing')

            with col3:
                shortness_of_breath = st.text_input('Shortness of Breath (0=No, 1=Yes)',
                                                  key='shortness_of_breath',
                                                  help='Difficulty breathing or shortness of breath')

            with col1:
                swallowing_difficulty = st.text_input('Swallowing Difficulty (0=No, 1=Yes)',
                                                    key='swallowing_difficulty',
                                                    help='Difficulty swallowing food or liquids')

            with col2:
                chest_pain = st.text_input('Chest Pain (0=No, 1=Yes)',
                                         key='chest_pain',
                                         help='Pain in the chest area')

            with col3:
                anxiety = st.text_input('Anxiety (0=No, 1=Yes)',
                                      key='anxiety',
                                      help='Feelings of anxiety or worry')

        # code for Prediction
        lung_cancer_diagnosis = ''

        # creating a button for Prediction
        if st.button('Lung Cancer Test Result'):
            try:
                # Get the column names from the original lung cancer dataset
                feature_names = X_lung_cancer.columns.tolist()

                # Create a list of input values
                input_values = [age, gender, smoking, yellow_fingers, anxiety, peer_pressure,
                              chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
                              shortness_of_breath, swallowing_difficulty, chest_pain]

                # Ensure the number of input values matches the number of features
                if len(input_values) != len(feature_names):
                    st.error(f"Error: Number of input values ({len(input_values)}) does not match number of features ({len(feature_names)})")
                else:
                    # Create a dictionary with feature names matching the training data
                    input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

                    # Convert to DataFrame with proper feature names
                    input_df = pd.DataFrame([input_dict])

                    # Convert string values to float
                    for column in input_df.columns:
                        input_df[column] = input_df[column].astype(float)

                    # Make prediction using DataFrame with feature names
                    lung_cancer_prediction = lung_cancer_model.predict(input_df)
                    
                    # Get probability scores (percentage chance of lung cancer)
                    lung_cancer_probability = lung_cancer_model.predict_proba(input_df)
                    # Check if the model has predicted probabilities for both classes
                    if lung_cancer_probability.shape[1] > 1:
                        lung_cancer_percentage = round(lung_cancer_probability[0][1] * 100, 2)  # Probability of class 1 (lung cancer)
                    else:
                        # If only one class was in the training data, use the prediction directly
                        lung_cancer_percentage = round(lung_cancer_probability[0][0] * 100, 2) if lung_cancer_prediction[0] == 1 else 0.0
                    
                    if lung_cancer_prediction[0] == 1:
                        lung_cancer_diagnosis = f"The person may have lung cancer (Risk: {lung_cancer_percentage}%)"
                        
                        # Personalized recommendations based on risk factors
                        recommendations = []
                        try:
                            smoking_value = float(smoking)
                            coughing_value = float(coughing)
                            shortness_value = float(shortness_of_breath)
                            chest_pain_value = float(chest_pain)
                            
                            if smoking_value == 1:
                                recommendations.append("• Quit smoking immediately. This is the most important step you can take.")
                            
                            if coughing_value == 1 or shortness_value == 1 or chest_pain_value == 1:
                                recommendations.append("• Your respiratory symptoms require immediate medical attention.")
                            
                            if float(age) > 60:
                                recommendations.append("• Your age is a significant risk factor. Early screening is crucial.")
                        except:
                            pass
                        
                        if not recommendations:
                            recommendations.append("• Seek immediate medical evaluation from a pulmonologist or oncologist.")
                        
                        st.warning(f"""
                        **What is Lung Cancer?**
                        Lung cancer is a type of cancer that begins in the lungs and is often related to smoking,
                        though it can occur in non-smokers as well. It is characterized by uncontrolled cell growth
                        in tissues of the lung, which can spread to other parts of the body.

                        **Your Risk Assessment: {lung_cancer_percentage}% risk of lung cancer**
                        
                        **Personalized Recommendations:**
                        {chr(10).join(recommendations)}
                        
                        **Next Steps:**
                        With a {lung_cancer_percentage}% risk assessment, we strongly recommend consulting with a healthcare 
                        provider immediately for proper evaluation and diagnosis. Early detection is crucial for effective 
                        treatment. This prediction is not a medical diagnosis.
                        """)
                    else:
                        lung_cancer_diagnosis = f"The person does not show indicators of lung cancer (Risk: {lung_cancer_percentage}%)"
                        st.success(lung_cancer_diagnosis)
                        
                        # Provide health advice even for negative predictions
                        recommendations = []
                        try:
                            smoking_value = float(smoking)
                            if smoking_value == 1:
                                recommendations.append("• While your current assessment doesn't indicate lung cancer, smoking is the leading risk factor. Quitting smoking can significantly reduce your risk over time.")
                        except:
                            pass
                        
                        st.info(f"""
                        **Your Risk Assessment: {lung_cancer_percentage}% risk of lung cancer**
                        
                        **Lung Health Recommendations:**
                        • Avoid tobacco smoke and other lung irritants
                        • Get regular exercise to maintain lung function
                        • Consider regular screenings if you have risk factors
                        • Be aware of persistent respiratory symptoms and report them to your doctor
                        
                        {chr(10).join(recommendations) if recommendations else ""}
                        
                        Even with a low risk assessment, maintaining lung health is important for prevention.
                        """)

            except ValueError:
                st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Hypothyroid Prediction Page
if (selected == 'Hypothyroid Prediction'):

    #page title
    st.title('Hypothyroid Prediction Using ML')

    if not hypothyroid_model_ready:
        st.error("Hypothyroid prediction is currently unavailable. Please add a hypothyroid.csv file to enable this feature.")
        st.info("The hypothyroid dataset should contain relevant features such as TSH levels, T3, T4, age, gender, etc., with a target column named 'Hypothyroid'.")
    else:
        st.write("""
        ### About Hypothyroidism
        Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones.
        The thyroid is a butterfly-shaped gland in the front of your neck that produces hormones that control 
        how your body uses energy and affects nearly every organ in your body.
        
        **Types of Hypothyroidism:**
        - **Primary Hypothyroidism**: The thyroid itself is not functioning properly
        - **Secondary Hypothyroidism**: The pituitary gland doesn't produce enough TSH to stimulate the thyroid
        - **Congenital Hypothyroidism**: Present at birth
        - **Subclinical Hypothyroidism**: Mildly elevated TSH with normal thyroid hormone levels
        
        **Common Symptoms:**
        - Fatigue and sluggishness
        - Increased sensitivity to cold
        - Constipation
        - Dry skin and hair
        - Unexplained weight gain
        - Puffy face
        - Hoarseness
        - Muscle weakness
        - Elevated cholesterol
        - Depression
        - Impaired memory
        
        **Risk Factors:**
        - Being female
        - Age over 60
        - Family history of thyroid disease
        - Autoimmune disorders (like Hashimoto's thyroiditis)
        - Previous thyroid surgery or radiation
        - Certain medications
        
        This prediction tool uses laboratory test results and patient information to assess the likelihood
        of hypothyroidism based on machine learning analysis of patient data.
        
        **Note:** This tool is for educational purposes only and should not replace professional medical advice.
        """)

        # Example values for hypothyroidism (High risk case)
        hypothyroid_example = {
            'hypo_age': '58',            # Older age increases risk
            'hypo_gender': '0',          # Female (higher risk)
            'tsh': '12.5',               # Significantly elevated (normal: 0.4-4.0)
            't3': '70',                  # Below normal range (normal: 80-200)
            't4': '3.8',                 # Below normal range (normal: 5.0-12.0)
            't4u': '0.7',                # Below normal range (normal: 0.8-1.2)
            'fti': '5.0',                # Below normal range (normal: 6.0-10.5)
            'on_thyroxine': '0',         # Not currently treated
            'on_antithyroid_meds': '0',  # No
            'sick': '1',                 # Other illness can affect thyroid
            'pregnant': '0',             # No
            'thyroid_surgery': '0',      # No
            'tumor': '1'                 # Thyroid tumor increases risk
        }

        # Add example values button at the top
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button('Fill Example Values (High Risk)', key='hypothyroid_example'):
                st.info("These values represent a high-risk case that is likely to show a positive prediction.")
                for key, value in hypothyroid_example.items():
                    st.session_state[key] = value

        # Create tabs for better organization
        tab1, tab2 = st.tabs(["Thyroid Tests", "Patient Information"])

        with tab1:
            st.subheader("Thyroid Function Tests")
            st.write("These are blood tests that measure thyroid hormone levels.")

            col1, col2 = st.columns(2)

            with col1:
                tsh = st.text_input('TSH Level (mIU/L)',
                                  key='tsh',
                                  help='Normal range: 0.4-4.0 mIU/L. Higher values may indicate hypothyroidism.')

                t3 = st.text_input('T3 Level (ng/dL)',
                                 key='t3',
                                 help='Normal range: 80-200 ng/dL. Lower values may indicate hypothyroidism.')

            with col2:
                t4 = st.text_input('T4 Level (μg/dL)',
                                 key='t4',
                                 help='Normal range: 5.0-12.0 μg/dL. Lower values may indicate hypothyroidism.')

                t4u = st.text_input('T4U Level',
                                  key='t4u',
                                  help='T4 Uptake Ratio. Normal range: 0.8-1.2.')

                fti = st.text_input('FTI Value',
                                  key='fti',
                                  help='Free Thyroxine Index. Normal range: 6.0-10.5.')

        with tab2:
            st.subheader("Patient Information")
            st.write("These are details about the patient's health history and demographics.")

            col1, col2 = st.columns(2)

            with col1:
                age = st.text_input('Age', key='hypo_age',
                                  help='Patient age in years')

                gender = st.text_input('Gender (0 for Female, 1 for Male)', key='hypo_gender',
                                     help='0 = Female, 1 = Male')

                on_thyroxine = st.text_input('On Thyroxine Medication (0 for No, 1 for Yes)',
                                           key='on_thyroxine',
                                           help='Is the patient currently taking thyroxine medication?')

                on_antithyroid_meds = st.text_input('On Antithyroid Medication (0 for No, 1 for Yes)',
                                                  key='on_antithyroid_meds',
                                                  help='Is the patient currently taking anti-thyroid medication?')

            with col2:
                sick = st.text_input('Currently Sick (0 for No, 1 for Yes)',
                                   key='sick',
                                   help='Is the patient currently sick with another illness?')

                pregnant = st.text_input('Pregnant (0 for No, 1 for Yes)',
                                       key='pregnant',
                                       help='Is the patient pregnant?')

                thyroid_surgery = st.text_input('Previous Thyroid Surgery (0 for No, 1 for Yes)',
                                              key='thyroid_surgery',
                                              help='Has the patient had thyroid surgery in the past?')

                tumor = st.text_input('Thyroid Tumor (0 for No, 1 for Yes)',
                                    key='tumor',
                                    help='Does the patient have a thyroid tumor?')

        # code for Prediction
        hypothyroid_diagnosis = ''

        # creating a button for Prediction
        if st.button('Hypothyroid Test Result'):
            try:
                # Get the column names from the original hypothyroid dataset
                feature_names = X_hypothyroid.columns.tolist()

                # Create a list of input values
                input_values = [age, gender, tsh, t3, t4, t4u, fti, on_thyroxine,
                              on_antithyroid_meds, sick, pregnant, thyroid_surgery, tumor]

                # Ensure the number of input values matches the number of features
                if len(input_values) != len(feature_names):
                    st.error(f"Error: Number of input values ({len(input_values)}) does not match number of features ({len(feature_names)})")
                else:
                    # Create a dictionary with feature names matching the training data
                    input_dict = {feature_names[i]: input_values[i] for i in range(len(feature_names))}

                    # Convert to DataFrame with proper feature names
                    input_df = pd.DataFrame([input_dict])

                    # Convert string values to float
                    for column in input_df.columns:
                        input_df[column] = input_df[column].astype(float)

                    # Make prediction using DataFrame with feature names
                    hypothyroid_prediction = hypothyroid_model.predict(input_df)
                    
                    # Get probability scores (percentage chance of hypothyroidism)
                    hypothyroid_probability = hypothyroid_model.predict_proba(input_df)
                    # Check if the model has predicted probabilities for both classes
                    if hypothyroid_probability.shape[1] > 1:
                        hypothyroid_percentage = round(hypothyroid_probability[0][1] * 100, 2)  # Probability of class 1 (hypothyroidism)
                    else:
                        # If only one class was in the training data, use the prediction directly
                        hypothyroid_percentage = round(hypothyroid_probability[0][0] * 100, 2) if hypothyroid_prediction[0] == 1 else 0.0
                    
                    if hypothyroid_prediction[0] == 1:
                        hypothyroid_diagnosis = f"The person may have hypothyroidism (Risk: {hypothyroid_percentage}%)"
                        
                        # Personalized recommendations based on thyroid values
                        recommendations = []
                        try:
                            tsh_value = float(tsh)
                            t4_value = float(t4)
                            
                            if tsh_value > 10.0:
                                recommendations.append("• Your TSH level is significantly elevated, indicating primary hypothyroidism.")
                            elif tsh_value > 4.0:
                                recommendations.append("• Your TSH level is elevated, suggesting mild hypothyroidism.")
                            
                            if t4_value < 5.0:
                                recommendations.append("• Your T4 level is below normal range, confirming hypothyroidism.")
                            
                            if float(age) > 60:
                                recommendations.append("• Hypothyroidism is more common in older adults and may present with subtle symptoms.")
                            
                            if float(sex) == 0:  # Female
                                recommendations.append("• Women are more likely to develop hypothyroidism. Regular monitoring is important.")
                        except:
                            pass
                        
                        if not recommendations:
                            recommendations.append("• Consult with an endocrinologist for proper evaluation and treatment options.")
                        
                        st.warning(f"""
                        **What is Hypothyroidism?**
                        Hypothyroidism is a condition where the thyroid gland doesn't produce enough thyroid hormones.
                        Common symptoms include fatigue, weight gain, cold intolerance, dry skin, and depression.

                        **Your Risk Assessment: {hypothyroid_percentage}% risk of hypothyroidism**
                        
                        **Personalized Insights:**
                        {chr(10).join(recommendations)}
                        
                        **Next Steps:**
                        With a {hypothyroid_percentage}% risk assessment, we recommend consulting with a healthcare provider
                        for proper diagnosis and treatment. Hypothyroidism is typically managed with thyroid hormone 
                        replacement therapy. This prediction is not a medical diagnosis.
                        """)
                    else:
                        hypothyroid_diagnosis = f"The person does not show indicators of hypothyroidism (Risk: {hypothyroid_percentage}%)"
                        st.success(hypothyroid_diagnosis)
                        
                        # Provide health advice even for negative predictions
                        recommendations = []
                        try:
                            tsh_value = float(tsh)
                            if tsh_value > 4.0:
                                recommendations.append("• Your TSH level appears to be elevated, which can sometimes indicate subclinical hypothyroidism. Consider discussing these results with your healthcare provider.")
                        except:
                            pass
                        
                        st.info(f"""
                        **Your Risk Assessment: {hypothyroid_percentage}% risk of hypothyroidism**
                        
                        **Thyroid Health Recommendations:**
                        • Ensure adequate iodine intake through diet (seafood, dairy, iodized salt)
                        • Be aware of symptoms like fatigue, weight gain, and cold intolerance
                        • Consider regular thyroid function tests if you have risk factors
                        • Discuss any family history of thyroid disorders with your doctor
                        
                        {chr(10).join(recommendations) if recommendations else ""}
                        
                        Even with a low risk assessment, monitoring thyroid health is important, especially if you have risk factors.
                        """)

            except ValueError:
                st.error("Please enter valid numerical values for all fields or use the 'Fill Example Values' button")
            except Exception as e:
                st.error(f"An error occurred: {e}")
