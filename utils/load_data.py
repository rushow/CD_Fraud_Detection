import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from datasets import load_dataset
from huggingface_hub import login as hf_login

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    hf_login(token=_hf_token, add_to_git_credential=False)

def _stratified_cap(X_vals, y, target_final=100000, target_ratio=1/3, rng=None):
    """
    Stratified sample sized so the final dataset after rebalancing is ~target_final rows.

    Low-fraud case (fraud rate < target_ratio):
        SMOTE will add synthetic minority samples → cap at target_final, final will
        be somewhat larger than target_final after oversampling.

    High-fraud case (fraud rate >= target_ratio):
        Undersampling shrinks the dataset.  We scale the cap up so that after
        removing excess fraud rows (and a ~20% ENN buffer) we still land above
        target_final.  Formula: need legit_count ≈ target_final * 3/4 * 1.2,
        so cap_total = legit_count / (1 - fraud_rate).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(y) <= target_final:
        return X_vals, y

    p = float((y == 1).mean())
    if p < target_ratio:
        cap_size = target_final
    else:
        # legit rows needed so that legit + legit/3 > target_final after ENN cleanup
        legit_needed = int(target_final * 0.75 * 1.20)   # 20% buffer for ENN
        cap_size = max(target_final, int(legit_needed / max(1.0 - p, 1e-9)))

    fraud_idx = np.where(y == 1)[0]
    legit_idx  = np.where(y == 0)[0]
    n_fraud_keep = max(10, int(cap_size * len(fraud_idx) / len(y)))
    n_legit_keep = min(cap_size - n_fraud_keep, len(legit_idx))
    n_fraud_keep = min(n_fraud_keep, len(fraud_idx))
    cap_idx = np.concatenate([
        rng.choice(fraud_idx, size=n_fraud_keep, replace=False),
        rng.choice(legit_idx, size=n_legit_keep, replace=False),
    ])
    return X_vals[cap_idx], y[cap_idx]


def load_synthetic_financial_data():
    """
    Load the Synthetic Financial Dataset For Fraud Detection
    """
    try:
        # Load from Hugging Face
        dataset = load_dataset("purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection", split="train")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)
        
        # Select a subset of features 
        feature_cols = [col for col in df.columns if col != 'isFraud' and col != 'isFlaggedFraud']
        
        # Prepare features and target
        X = df[feature_cols].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Get target variable
        y = df['isFraud'].values

        # Fill any NaNs before SMOTE
        X = X.fillna(X.mean(numeric_only=True))

        feature_cols_ordered = X.columns.tolist()
        X_vals = X.values
        rng = np.random.default_rng(42)
        X_vals, y = _stratified_cap(X_vals, y, target_final=100000, target_ratio=0.5, rng=rng)
        _n_fraud_pre = int((y == 1).sum())
        _n_legit_pre = int((y == 0).sum())
        print(f"Pre-SMOTE — synthetic_financial: {_n_fraud_pre} fraud, {_n_legit_pre} legitimate (from original {len(df):,} rows)")

        # Apply SMOTE+ENN: SMOTE oversamples fraud to reach a 1:2 ratio
        # (sampling_strategy=0.5 → minority/majority = 0.5), then ENN removes
        # noisy/borderline samples from both classes, giving a smaller, cleaner dataset.
        sm_enn = SMOTEENN(sampling_strategy=0.5, random_state=42)
        X_arr, y_arr = sm_enn.fit_resample(X_vals, y)

        n_fraud = int((y_arr == 1).sum())
        n_legit = int((y_arr == 0).sum())
        print(f"After SMOTE+ENN — synthetic_financial: {n_fraud} fraud, {n_legit} legitimate (ratio ~1:2)")

        # Shuffle so synthetic and real samples are interleaved
        rng = np.random.default_rng(42)
        order = rng.permutation(len(y_arr))
        X_arr, y_arr = X_arr[order], y_arr[order]

        # Convert to dictionary format for River
        X = pd.DataFrame(X_arr, columns=feature_cols_ordered).to_dict(orient='records')
        y = y_arr

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

        # Get target variable
        y = df[target_col].values

        X_vals = X.values
        rng = np.random.default_rng(42)
        X_vals, y = _stratified_cap(X_vals, y, target_final=100000, target_ratio=1/3, rng=rng)
        _n_fraud_pre = int((y == 1).sum())
        _n_legit_pre = int((y == 0).sum())
        print(f"Pre-SMOTE — nooha_cc_fraud: {_n_fraud_pre} fraud, {_n_legit_pre} legitimate (from original {len(df):,} rows)")

        current_ratio = _n_fraud_pre / _n_legit_pre if _n_legit_pre > 0 else 1.0
        feature_cols_ordered = X.columns.tolist()

        if current_ratio >= 1/3:
            # Undersample fraud to ~1:3, then ENN removes borderline samples naturally.
            target_n_fraud = max(1, _n_legit_pre // 3)
            fraud_all_idx  = np.where(y == 1)[0]
            legit_all_idx  = np.where(y == 0)[0]
            sel_fraud_idx  = rng.choice(fraud_all_idx, size=min(target_n_fraud, len(fraud_all_idx)), replace=False)
            combined_idx   = np.concatenate([sel_fraud_idx, legit_all_idx])
            rng.shuffle(combined_idx)
            X_tmp, y_tmp = X_vals[combined_idx], y[combined_idx]
            _enn = EditedNearestNeighbours(n_jobs=-1)
            X_arr, y_arr = _enn.fit_resample(X_tmp, y_tmp)
        else:
            # Apply SMOTE+ENN: SMOTE oversamples fraud to 1:3 ratio, then ENN removes
            # noisy/borderline samples from both classes to reduce and clean the dataset.
            _smote = SMOTE(sampling_strategy=1/3, random_state=42)
            _enn   = EditedNearestNeighbours(n_jobs=-1)
            sm_enn = SMOTEENN(smote=_smote, enn=_enn)
            X_arr, y_arr = sm_enn.fit_resample(X_vals, y)

        n_fraud = int((y_arr == 1).sum())
        n_legit = int((y_arr == 0).sum())
        print(f"After rebalancing — nooha_cc_fraud: {n_fraud} fraud, {n_legit} legitimate")

        # Shuffle so synthetic and real fraud samples are interleaved
        rng = np.random.default_rng(42)
        order = rng.permutation(len(y_arr))
        X_arr, y_arr = X_arr[order], y_arr[order]

        # Convert to dictionary format for River
        X = pd.DataFrame(X_arr, columns=feature_cols_ordered).to_dict(orient='records')
        y = y_arr

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
            for col in df.columns:
                if df[col].nunique() == 2 and df[col].dtype in ['int64', 'int32', 'bool']:
                    target_col = col
                    break
            else:
                target_col = df.columns[-1]

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

        # Get target variable
        y = df[target_col].values

        X_vals = X.values
        rng = np.random.default_rng(42)
        X_vals, y = _stratified_cap(X_vals, y, target_final=100000, target_ratio=1/3, rng=rng)
        _n_fraud_pre = int((y == 1).sum())
        _n_legit_pre = int((y == 0).sum())
        print(f"Pre-SMOTE — european_cc_fraud: {_n_fraud_pre} fraud, {_n_legit_pre} legitimate (from original {len(df):,} rows)")

        current_ratio = _n_fraud_pre / _n_legit_pre if _n_legit_pre > 0 else 1.0
        feature_cols_ordered = X.columns.tolist()

        if current_ratio >= 1/3:
            # Undersample fraud to ~1:3, then ENN removes borderline samples naturally.
            target_n_fraud = max(1, _n_legit_pre // 3)
            fraud_all_idx = np.where(y == 1)[0]
            legit_all_idx = np.where(y == 0)[0]
            sel_fraud_idx = rng.choice(fraud_all_idx, size=min(target_n_fraud, len(fraud_all_idx)), replace=False)
            combined_idx  = np.concatenate([sel_fraud_idx, legit_all_idx])
            rng.shuffle(combined_idx)
            X_tmp, y_tmp = X_vals[combined_idx], y[combined_idx]
            _enn = EditedNearestNeighbours(n_jobs=-1)
            X_arr, y_arr = _enn.fit_resample(X_tmp, y_tmp)
        else:
            # Apply SMOTE+ENN: SMOTE oversamples fraud to 1:3 ratio, then ENN removes
            # noisy/borderline samples from both classes to reduce and clean the dataset.
            _smote = SMOTE(sampling_strategy=1/3, random_state=42)
            _enn   = EditedNearestNeighbours(n_jobs=-1)
            sm_enn = SMOTEENN(smote=_smote, enn=_enn)
            X_arr, y_arr = sm_enn.fit_resample(X_vals, y)

        n_fraud = int((y_arr == 1).sum())
        n_legit = int((y_arr == 0).sum())
        print(f"After rebalancing — european_cc_fraud: {n_fraud} fraud, {n_legit} legitimate")

        # Shuffle so synthetic and real fraud samples are interleaved
        order = rng.permutation(len(y_arr))
        X_arr, y_arr = X_arr[order], y_arr[order]

        # Convert to dictionary format for River
        X = pd.DataFrame(X_arr, columns=feature_cols_ordered).to_dict(orient='records')
        y = y_arr

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

        # Get target variable
        y = df[target_col].values

        X_vals = X.values
        rng = np.random.default_rng(42)
        X_vals, y = _stratified_cap(X_vals, y, target_final=100000, target_ratio=1/3, rng=rng)
        _n_fraud_pre = int((y == 1).sum())
        _n_legit_pre = int((y == 0).sum())
        print(f"Pre-cap — thomask_cc_fraud: {_n_fraud_pre} fraud, {_n_legit_pre} legitimate (from original {len(df):,} rows)")

        current_ratio = _n_fraud_pre / _n_legit_pre if _n_legit_pre > 0 else 1.0
        feature_cols_ordered = X.columns.tolist()

        if current_ratio >= 1/3:
            # Undersample fraud to ~1:3, then ENN removes borderline samples naturally.
            target_n_fraud = max(1, _n_legit_pre // 3)
            fraud_all_idx  = np.where(y == 1)[0]
            legit_all_idx  = np.where(y == 0)[0]
            sel_fraud_idx  = rng.choice(fraud_all_idx, size=min(target_n_fraud, len(fraud_all_idx)), replace=False)
            combined_idx   = np.concatenate([sel_fraud_idx, legit_all_idx])
            rng.shuffle(combined_idx)
            X_tmp, y_tmp = X_vals[combined_idx], y[combined_idx]
            _enn = EditedNearestNeighbours(n_jobs=-1)
            X_arr, y_arr = _enn.fit_resample(X_tmp, y_tmp)
        else:
            # Apply SMOTE+ENN: SMOTE oversamples fraud to 1:3 ratio, then ENN removes
            # noisy/borderline samples from both classes to reduce and clean the dataset.
            _smote = SMOTE(sampling_strategy=1/3, random_state=42)
            _enn   = EditedNearestNeighbours(n_jobs=-1)
            sm_enn = SMOTEENN(smote=_smote, enn=_enn)
            X_arr, y_arr = sm_enn.fit_resample(X_vals, y)

        n_fraud = int((y_arr == 1).sum())
        n_legit = int((y_arr == 0).sum())
        print(f"After rebalancing — thomask_cc_fraud: {n_fraud} fraud, {n_legit} legitimate")

        # Shuffle so synthetic and real fraud samples are interleaved
        rng = np.random.default_rng(42)
        order = rng.permutation(len(y_arr))
        X_arr, y_arr = X_arr[order], y_arr[order]

        # Convert to dictionary format for River
        X = pd.DataFrame(X_arr, columns=feature_cols_ordered).to_dict(orient='records')
        y = y_arr

        return "thomask_cc_fraud", X, y

    except Exception as e:
        print(f"Error loading thomask CC fraud data: {str(e)}")
        return "thomask_cc_fraud", [], []

# def load_bank_transaction_fraud_data():
#     """
#     Load the Bank Transaction Fraud Dataset
#     """
#     try:
#         # Load from Hugging Face
#         dataset = load_dataset("qppd/bank-transaction-fraud", split="train")
        
#         # Convert to pandas DataFrame
#         df = pd.DataFrame(dataset)
        
#         # Identify target column
#         if 'is_fraud' in df.columns:
#             target_col = 'is_fraud'
#         elif 'fraud' in df.columns:
#             target_col = 'fraud'
#         elif 'isFraud' in df.columns:
#             target_col = 'isFraud'
#         else:
#             # Find binary column that's likely to be the target
#             for col in df.columns:
#                 if df[col].nunique() == 2 and df[col].dtype in ['int64', 'int32', 'bool']:
#                     target_col = col
#                     break
#             else:
#                 target_col = df.columns[-1]  # Default to last column
        
#         # Exclude target from features
#         feature_cols = [col for col in df.columns if col != target_col]
        
#         # Prepare features
#         X = df[feature_cols].copy()
        
#         # Handle categorical variables
#         categorical_cols = X.select_dtypes(include=['object']).columns
#         for col in categorical_cols:
#             le = LabelEncoder()
#             X[col] = le.fit_transform(X[col].astype(str))
        
#         # Fill NaN values
#         X = X.fillna(X.mean())
        
#         # Convert to dictionary format for River
#         X = X.to_dict(orient='records')
        
#         # Get target variable
#         y = df[target_col].values
        
#         return "bank_transaction_fraud", X, y
    
#     except Exception as e:
#         print(f"Error loading bank transaction fraud data: {str(e)}")
#         return "bank_transaction_fraud", [], []

# def load_cifer_fraud_detection_data():
#     """
#     Load the Cifer Fraud Detection Dataset (AF)
#     """
#     try:
#         dataset = load_dataset("CiferAI/Cifer-Fraud-Detection-Dataset-AF", split="train")

#         df = pd.DataFrame(dataset)

#         target_col = None
#         preferred_targets = [
#             "label",
#             "Class",
#             "class",
#             "isFraud",
#             "is_fraud",
#             "fraud",
#             "target",
#             "y",
#         ]
#         for col in preferred_targets:
#             if col in df.columns:
#                 target_col = col
#                 break

#         if target_col is None:
#             for col in df.columns:
#                 if df[col].nunique() == 2 and df[col].dtype in [
#                     "int64",
#                     "int32",
#                     "int16",
#                     "int8",
#                     "uint8",
#                     "bool",
#                 ]:
#                     target_col = col
#                     break

#         if target_col is None:
#             target_col = df.columns[-1]

#         feature_cols = [col for col in df.columns if col != target_col]

#         X = df[feature_cols].copy()

#         categorical_cols = X.select_dtypes(include=["object"]).columns
#         for col in categorical_cols:
#             le = LabelEncoder()
#             X[col] = le.fit_transform(X[col].astype(str))

#         numeric_cols = X.select_dtypes(include=["number"]).columns
#         if len(numeric_cols) > 0:
#             X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

#         X = X.fillna(0)

#         X = X.to_dict(orient="records")

#         y = df[target_col].values

#         return "cifer_fraud_detection_af", X, y

#     except Exception as e:
#         print(f"Error loading Cifer fraud detection data: {str(e)}")
#         return "cifer_fraud_detection_af", [], []

# def load_nigerian_financial_fraud_data():
#     """
#     Load the Nigerian Financial Transactions and Fraud Detection Dataset
#     """
#     try:
#         dataset = load_dataset(
#             "electricsheepafrica/Nigerian-Financial-Transactions-and-Fraud-Detection-Dataset",
#             split="train",
#         )

#         df = pd.DataFrame(dataset)

#         target_col = None
#         preferred_targets = [
#             "label",
#             "Class",
#             "class",
#             "isFraud",
#             "is_fraud",
#             "fraud",
#             "target",
#             "y",
#         ]
#         for col in preferred_targets:
#             if col in df.columns:
#                 target_col = col
#                 break

#         if target_col is None:
#             for col in df.columns:
#                 if df[col].nunique() == 2 and df[col].dtype in [
#                     "int64",
#                     "int32",
#                     "int16",
#                     "int8",
#                     "uint8",
#                     "bool",
#                 ]:
#                     target_col = col
#                     break

#         if target_col is None:
#             target_col = df.columns[-1]

#         feature_cols = [col for col in df.columns if col != target_col]

#         X = df[feature_cols].copy()

#         categorical_cols = X.select_dtypes(include=["object"]).columns
#         for col in categorical_cols:
#             le = LabelEncoder()
#             X[col] = le.fit_transform(X[col].astype(str))

#         numeric_cols = X.select_dtypes(include=["number"]).columns
#         if len(numeric_cols) > 0:
#             X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

#         X = X.fillna(0)

#         X = X.to_dict(orient="records")

#         y = df[target_col].values

#         return "nigerian_financial_fraud", X, y

#     except Exception as e:
#         print(f"Error loading Nigerian financial fraud data: {str(e)}")
#         return "nigerian_financial_fraud", [], []

# def load_amitkedia_financial_fraud_data():
#     """
#     Load the Financial Fraud Dataset by amitkedia
#     """
#     try:
#         dataset = load_dataset("amitkedia/Financial-Fraud-Dataset", split="train")

#         df = pd.DataFrame(dataset)

#         target_col = None
#         preferred_targets = [
#             "label",
#             "Class",
#             "class",
#             "isFraud",
#             "is_fraud",
#             "fraud",
#             "target",
#             "y",
#         ]
#         for col in preferred_targets:
#             if col in df.columns:
#                 target_col = col
#                 break

#         if target_col is None:
#             for col in df.columns:
#                 if df[col].nunique() == 2 and df[col].dtype in [
#                     "int64",
#                     "int32",
#                     "int16",
#                     "int8",
#                     "uint8",
#                     "bool",
#                 ]:
#                     target_col = col
#                     break

#         if target_col is None:
#             target_col = df.columns[-1]

#         feature_cols = [col for col in df.columns if col != target_col]

#         X = df[feature_cols].copy()

#         categorical_cols = X.select_dtypes(include=["object"]).columns
#         for col in categorical_cols:
#             le = LabelEncoder()
#             X[col] = le.fit_transform(X[col].astype(str))

#         numeric_cols = X.select_dtypes(include=["number"]).columns
#         if len(numeric_cols) > 0:
#             X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

#         X = X.fillna(0)

#         X = X.to_dict(orient="records")

#         y = df[target_col].values

#         return "amitkedia_financial_fraud", X, y

#     except Exception as e:
#         print(f"Error loading amitkedia financial fraud data: {str(e)}")
#         return "amitkedia_financial_fraud", [], []


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