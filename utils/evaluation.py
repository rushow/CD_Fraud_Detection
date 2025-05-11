from river import metrics
import numpy as np
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

def evaluate_learner(df_name, model, drift_detector, X, y):

    # Initialize metrics
    metric = metrics.Accuracy()  # Accuracy metric
    metric_f1 = metrics.F1()  # F1 score
    metric_precision = metrics.Precision()  # Precision score

    t = []  # Number of evaluated data points
    m = []  # Real-time accuracy

    yt = []  # True labels
    yp = []  # Predicted labels
    yp_proba = []  # Predicted probabilities
    
    # Check if we have data to evaluate
    if len(X) == 0 or len(y) == 0:
        print(f"Warning: No data available for {df_name}")
        return [], [], 0, 0, 0, None, None

    # Process each instance one by one (streaming fashion)
    for i, (x, y_true) in enumerate(zip(X, y)):
        # Make prediction
        try:
            y_pred = model.predict_one(x)
            y_proba = model.predict_proba_one(x)
        except Exception as e:
            print(f"Error predicting at instance {i}: {str(e)}")
            continue
            
        # Learn from this instance
        try:
            model.learn_one(x, y_true)
        except Exception as e:
            print(f"Error learning from instance {i}: {str(e)}")
            continue

        # Update the drift detector
        try:
            # Different detectors may require different update methods
            if hasattr(drift_detector, 'update'):
                drift_detector.update(x, y_true)
            elif hasattr(drift_detector, 'update_data'):
                drift_detector.update_data(y_pred, y_true)
            else:
                # Default fallback
                drift_detector.update(y_pred, y_true)
                
            # Check for drift
            if drift_detector.drift_detected:
                print(f"Drift detected at instance {i}")
                try:
                    model = model.clone()  # Reset the model
                except Exception as e:
                    print(f"Error cloning model: {str(e)}")
        except Exception as e:
            print(f"Error in drift detection at instance {i}: {str(e)}")
            continue

        # Update metrics
        metric.update(y_true, y_pred)
        metric_f1.update(y_true, y_pred)
        metric_precision.update(y_true, y_pred)

        t.append(i)
        m.append(metric.get() * 100)

        yt.append(y_true)
        yp.append(y_pred)
        
        # Extract probability for class 1
        class_1_proba = y_proba.get(1, 0.5) if isinstance(y_proba, dict) and y_proba is not None else 0.5
        yp_proba.append(class_1_proba)
        
        # Print progress every 1000 instances
        # if i > 0 and i % 1000 == 0:
        #     print(f"Processed {i} instances, current accuracy: {metric.get():.4f}")

    # Compute AUC and AUROC if we have at least two classes
    if len(set(yt)) > 1:
        try:
            auroc_value = roc_auc_score(yt, yp_proba)
            precision, recall, _ = precision_recall_curve(yt, yp_proba)
            auc_value = auc(recall, precision)
        except Exception as e:
            print(f"Error computing AUC/AUROC: {str(e)}")
            auroc_value = None
            auc_value = None
    else:
        auroc_value = None
        auc_value = None
        print("Warning: Only one class present in y_true. ROC AUC score and AUC value are not defined in that case.")

    return t, m, metric, metric_f1, metric_precision, auc_value, auroc_value