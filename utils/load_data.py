import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import load_dataset
import os

def load_synthetic_financial_data():
    """
    Load the Synthetic Financial Dataset For Fraud Detection
    """
    try:
        # Load from Hugging Face
        dataset = load_dataset("purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection", split="train")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Select a subset of features (adjust based on the actual columns in the dataset)
        feature_cols = [col for col in df.columns if col != 'isFraud' and col != 'isFlaggedFraud']
        
        # Prepare features and target
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert to dictionary format for River
        X = X.to_dict(orient='records')
        
        # Get target variable
        y = df['isFraud'].values
        
        return "synthetic_financial", X, y
    
    except Exception as e:
        print(f"Error loading synthetic financial data: {str(e)}")
        # Return minimal data to continue execution
        return "synthetic_financial", [], []

def load_nooha_cc_fraud_data():
    """
    Load the Credit Card Fraud Detection Dataset from Nooha
    """
    try:
        # Load from Hugging Face
        dataset = load_dataset("Nooha/cc_fraud_detection_dataset", split="train")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Select relevant features and target
        if 'Class' in df.columns:
            target_col = 'Class'
        elif 'isFraud' in df.columns:
            target_col = 'isFraud'
        else:
            # Find the target column which should be binary
            for col in df.columns:
                if df[col].nunique() == 2 and df[col].dtype in ['int64', 'int32', 'bool']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]  # Default to last column if no suitable binary column found
        
        # Exclude target from features
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
        # Fill NaN values
        X = X.fillna(X.mean())
        
        # Convert to dictionary format for River
        X = X.to_dict(orient='records')
        
        # Get target variable
        y = df[target_col].values
        
        return "nooha_cc_fraud", X, y
    
    except Exception as e:
        print(f"Error loading Nooha CC fraud data: {str(e)}")
        return "nooha_cc_fraud", [], []

def load_european_cc_fraud_data():
    """
    Load the European Credit Card Fraud Dataset
    """
    try:
        # Load from Hugging Face
        dataset = load_dataset("stanpony/european_credit_card_fraud_dataset", split="train")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Identify target column (Class or similar)
        if 'Class' in df.columns:
            target_col = 'Class'
        else:
            # Find binary column that's likely to be the target
            for col in df.columns:
                if df[col].nunique() == 2 and df[col].dtype in ['int64', 'int32', 'bool']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]  # Default to last column
        
        # Exclude target from features
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Fill NaN values
        X = X.fillna(X.mean())
        
        # Convert to dictionary format for River
        X = X.to_dict(orient='records')
        
        # Get target variable
        y = df[target_col].values
        
        return "european_cc_fraud", X, y
    
    except Exception as e:
        print(f"Error loading European CC fraud data: {str(e)}")
        return "european_cc_fraud", [], []

def load_thomask_cc_fraud_data():
    """
    Load the Credit Card Fraud Dataset from thomask1018
    """
    try:
        # Load from Hugging Face
        dataset = load_dataset("thomask1018/credit_card_fraud", split="train")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Identify target column
        if 'Class' in df.columns:
            target_col = 'Class'
        elif 'isFraud' in df.columns:
            target_col = 'isFraud'
        else:
            # Find binary column that's likely to be the target
            for col in df.columns:
                if df[col].nunique() == 2 and df[col].dtype in ['int64', 'int32', 'bool']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]  # Default to last column
        
        # Exclude target from features
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill NaN values
        X = X.fillna(X.mean())
        
        # Convert to dictionary format for River
        X = X.to_dict(orient='records')
        
        # Get target variable
        y = df[target_col].values
        
        return "thomask_cc_fraud", X, y
    
    except Exception as e:
        print(f"Error loading thomask CC fraud data: {str(e)}")
        return "thomask_cc_fraud", [], []

def load_bank_transaction_fraud_data():
    """
    Load the Bank Transaction Fraud Dataset
    """
    try:
        # Load from Hugging Face
        dataset = load_dataset("qppd/bank-transaction-fraud", split="train")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Identify target column
        if 'is_fraud' in df.columns:
            target_col = 'is_fraud'
        elif 'fraud' in df.columns:
            target_col = 'fraud'
        elif 'isFraud' in df.columns:
            target_col = 'isFraud'
        else:
            # Find binary column that's likely to be the target
            for col in df.columns:
                if df[col].nunique() == 2 and df[col].dtype in ['int64', 'int32', 'bool']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]  # Default to last column
        
        # Exclude target from features
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Prepare features
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill NaN values
        X = X.fillna(X.mean())
        
        # Convert to dictionary format for River
        X = X.to_dict(orient='records')
        
        # Get target variable
        y = df[target_col].values
        
        return "bank_transaction_fraud", X, y
    
    except Exception as e:
        print(f"Error loading bank transaction fraud data: {str(e)}")
        return "bank_transaction_fraud", [], []

# Helper function to subsample data for faster processing
def subsample_data(X, y, max_samples=10000, random_state=42):
    """
    Subsample data to a more manageable size while preserving class distribution
    """
    n_samples = len(y)
    
    if n_samples <= max_samples:
        return X, y
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get indices of positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Calculate ratio of positive samples
    pos_ratio = len(pos_indices) / n_samples
    
    # Calculate number of positive and negative samples to select
    n_pos_samples = int(max_samples * pos_ratio)
    n_neg_samples = max_samples - n_pos_samples
    
    # Ensure we don't select more than available
    n_pos_samples = min(n_pos_samples, len(pos_indices))
    n_neg_samples = min(n_neg_samples, len(neg_indices))
    
    # Randomly select samples
    selected_pos_indices = np.random.choice(pos_indices, size=n_pos_samples, replace=False)
    selected_neg_indices = np.random.choice(neg_indices, size=n_neg_samples, replace=False)
    
    # Combine indices
    selected_indices = np.concatenate([selected_pos_indices, selected_neg_indices])
    
    # Shuffle indices
    np.random.shuffle(selected_indices)
    
    # Select samples
    X_subsample = [X[i] for i in selected_indices]
    y_subsample = y[selected_indices]
    
    return X_subsample, y_subsample