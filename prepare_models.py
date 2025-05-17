import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import os

# Create a directory for saved models if it doesn't exist
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# Function to train and save a model
def train_and_save_model(data_path, target_column, model_name, drop_columns=None):
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Drop non-feature columns if specified
        if drop_columns:
            data = data.drop(drop_columns, axis=1)
        
        # Split features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the model
        with open(f'saved_models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model {model_name} trained and saved successfully!")
        return True
    
    except Exception as e:
        print(f"Error training and saving {model_name} model: {e}")
        return False

# Train and save diabetes model
diabetes_success = train_and_save_model(
    data_path='diabetes.csv',
    target_column='Outcome',
    model_name='diabetes_model'
)

# Train and save heart disease model
heart_success = train_and_save_model(
    data_path='heart.csv',
    target_column='target',  # Assuming the target column is named 'target'
    model_name='heart_disease_model'
)

# Train and save Parkinson's model
parkinsons_success = train_and_save_model(
    data_path='parkinsons.csv',
    target_column='status',
    drop_columns=['name'],
    model_name='parkinsons_model'
)

# Train and save lung cancer model if the dataset exists
try:
    lung_cancer_success = train_and_save_model(
        data_path='lung_cancer.csv',
        target_column='Cancer',
        model_name='lung_cancer_model'
    )
except FileNotFoundError:
    print("Lung cancer dataset not found. Skipping model training.")
    lung_cancer_success = False

# Train and save hypothyroid model if the dataset exists
try:
    hypothyroid_success = train_and_save_model(
        data_path='hypothyroid.csv',
        target_column='Hypothyroid',
        model_name='hypothyroid_model'
    )
except FileNotFoundError:
    print("Hypothyroid dataset not found. Skipping model training.")
    hypothyroid_success = False

# Print summary
print("\nModel Training Summary:")
print(f"Diabetes Model: {'Success' if diabetes_success else 'Failed'}")
print(f"Heart Disease Model: {'Success' if heart_success else 'Failed'}")
print(f"Parkinson's Model: {'Success' if parkinsons_success else 'Failed'}")
print(f"Lung Cancer Model: {'Success' if lung_cancer_success else 'Failed'}")
print(f"Hypothyroid Model: {'Success' if hypothyroid_success else 'Failed'}")