import math

class EWMADriftDetector:
    def __init__(self, min_instances=30, lambda_=0.2, window_size=50):

        self.min_instances = min_instances
        self.lambda_ = lambda_
        self.window_size = window_size
        self.reset()

    def reset(self):
        """
        Reset the EWMA Drift Detector.
        """
        self.num_instances = 0
        self.drift_detected = False
        # Rolling buffer of 2*window_size errors; first half = reference, second = current
        self._error_buffer = []
        self.window1 = []
        self.window2 = []
        self.window1_error_sum = 0
        self.window2_error_sum = 0

    def update(self, y_pred, y_true):
        """
        Update the EWMA with new prediction result and check for drift.

        Parameters:
        - y_pred: Predicted value (binary).
        - y_true: True value (binary).

        Returns:
        - 'drift' if drift is detected, 'no_drift' otherwise.
        """
        error = 1 if y_pred != y_true else 0

        self.num_instances += 1

        # Maintain a rolling buffer so window1 (older half) and window2 (newer half)
        # always contain DIFFERENT samples.
        self._error_buffer.append(error)
        if len(self._error_buffer) > 2 * self.window_size:
            self._error_buffer.pop(0)

        if len(self._error_buffer) < 2 * self.window_size:
            return 'no_drift'

        # Split buffer: older half = reference, newer half = current
        mid = self.window_size
        self.window1 = self._error_buffer[:mid]
        self.window2 = self._error_buffer[mid:]
        self.window1_error_sum = sum(self.window1)
        self.window2_error_sum = sum(self.window2)
        
        # Calculate statistics for window1 (reference window)
        window1_p = self.window1_error_sum / len(self.window1)
        
        # Calculate EWMA for window1
        z_t1 = 0.0
        for err in self.window1:
            z_t1 += self.lambda_ * (err - z_t1)
        
        # Calculate standard deviation for window1
        window1_s = math.sqrt(
            window1_p * (1.0 - window1_p) * self.lambda_ * 
            (1.0 - math.pow(1.0 - self.lambda_, 2.0 * len(self.window1))) / (2.0 - self.lambda_)
        )
        
        # Calculate statistics for window2 (current window)
        window2_p = self.window2_error_sum / len(self.window2)
        
        # Calculate EWMA for window2
        z_t2 = 0.0
        for err in self.window2:
            z_t2 += self.lambda_ * (err - z_t2)
        
        # Calculate standard deviation for window2
        window2_s = math.sqrt(
            window2_p * (1.0 - window2_p) * self.lambda_ * 
            (1.0 - math.pow(1.0 - self.lambda_, 2.0 * len(self.window2))) / (2.0 - self.lambda_)
        )

        # Calculate the L_t control limit based on window1
        L_t = (
            3.97 - 6.56 * window1_p + 48.73 * math.pow(window1_p, 3) - 
            330.13 * math.pow(window1_p, 5) + 848.18 * math.pow(window1_p, 7)
        )

        # Detect drift or warning by comparing window2 against window1
        if self.num_instances < self.min_instances:
            return 'no_drift'

        if z_t2 > window1_p + L_t * window1_s:
            self.drift_detected = True
            self.reset()  # Reset after drift detection
            return 'drift'
        elif z_t2 > window1_p + 0.5 * L_t * window1_s:
            return 'warning'
        else:
            self.drift_detected = False
            return 'no_drift'

