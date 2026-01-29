"""
Model Monitoring & Metrics
===========================

Tracks prediction quality, model performance, and detects model drift.

Features:
- Prometheus metrics export
- Prediction quality tracking
- Model drift detection
- Alert generation
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prometheus Metrics
prediction_counter = Counter(
    'energy_predictions_total',
    'Total number of predictions made',
    ['energy_type', 'model']
)

prediction_latency = Histogram(
    'energy_prediction_latency_seconds',
    'Time taken to generate predictions',
    ['energy_type']
)

prediction_mae = Gauge(
    'energy_prediction_mae',
    'Mean Absolute Error of recent predictions',
    ['energy_type', 'window']
)

prediction_mape = Gauge(
    'energy_prediction_mape',
    'Mean Absolute Percentage Error',
    ['energy_type', 'window']
)

model_drift_score = Gauge(
    'energy_model_drift_score',
    'Model drift detection score (0-1, higher = more drift)',
    ['energy_type']
)

data_quality_score = Gauge(
    'energy_data_quality_score',
    'Data quality score (0-1, higher = better)',
    ['energy_type']
)

api_requests_total = Counter(
    'energy_api_requests_total',
    'Total API requests',
    ['endpoint', 'status']
)


@dataclass
class PredictionRecord:
    """Record of a single prediction"""
    timestamp: datetime
    energy_type: str
    actual_value: Optional[float]
    predicted_value: float
    horizon_hours: int
    model_name: str
    
    @property
    def error(self) -> Optional[float]:
        if self.actual_value is None:
            return None
        return abs(self.actual_value - self.predicted_value)
    
    @property
    def percentage_error(self) -> Optional[float]:
        if self.actual_value is None or self.actual_value == 0:
            return None
        return abs(self.actual_value - self.predicted_value) / self.actual_value * 100


class ModelMonitor:
    """Monitor model performance and detect drift"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: Dict[str, List[PredictionRecord]] = {}
        self.baseline_metrics: Dict[str, Dict] = {}
        logger.info(f"Model Monitor initialized (window_size={window_size})")
    
    def record_prediction(self, record: PredictionRecord):
        """Record a new prediction"""
        if record.energy_type not in self.predictions:
            self.predictions[record.energy_type] = []
        
        self.predictions[record.energy_type].append(record)
        
        # Keep only recent predictions
        if len(self.predictions[record.energy_type]) > self.window_size:
            self.predictions[record.energy_type] = \
                self.predictions[record.energy_type][-self.window_size:]
        
        # Update Prometheus metrics
        prediction_counter.labels(
            energy_type=record.energy_type,
            model=record.model_name
        ).inc()
        
        # Update error metrics if actual value is available
        if record.actual_value is not None:
            self._update_error_metrics(record.energy_type)
    
    def _update_error_metrics(self, energy_type: str):
        """Update error metrics for recent predictions"""
        records = [r for r in self.predictions[energy_type] if r.actual_value is not None]
        
        if not records:
            return
        
        # Calculate MAE for different windows
        for window in [10, 50, 100]:
            recent = records[-window:]
            if recent:
                mae = np.mean([r.error for r in recent if r.error is not None])
                prediction_mae.labels(
                    energy_type=energy_type,
                    window=f"{window}_predictions"
                ).set(mae)
                
                mape = np.mean([r.percentage_error for r in recent 
                               if r.percentage_error is not None])
                if not np.isnan(mape):
                    prediction_mape.labels(
                        energy_type=energy_type,
                        window=f"{window}_predictions"
                    ).set(mape)
    
    def set_baseline_metrics(self, energy_type: str, metrics: Dict):
        """Set baseline metrics for drift detection"""
        self.baseline_metrics[energy_type] = {
            'mae': metrics.get('mae', 0),
            'mape': metrics.get('mape', 0),
            'r2': metrics.get('r2', 0),
            'timestamp': datetime.now()
        }
        logger.info(f"Baseline metrics set for {energy_type}: {metrics}")
    
    def detect_drift(self, energy_type: str) -> Dict:
        """
        Detect model drift by comparing current performance to baseline
        
        Returns:
            Dict with drift_detected (bool) and drift_score (0-1)
        """
        if energy_type not in self.baseline_metrics:
            return {'drift_detected': False, 'drift_score': 0.0, 'reason': 'No baseline'}
        
        if energy_type not in self.predictions or len(self.predictions[energy_type]) < 20:
            return {'drift_detected': False, 'drift_score': 0.0, 'reason': 'Insufficient data'}
        
        # Get recent predictions with actual values
        records = [r for r in self.predictions[energy_type][-100:] if r.actual_value is not None]
        
        if len(records) < 10:
            return {'drift_detected': False, 'drift_score': 0.0, 'reason': 'Insufficient actuals'}
        
        # Calculate current metrics
        current_mae = np.mean([r.error for r in records if r.error is not None])
        current_mape = np.mean([r.percentage_error for r in records 
                               if r.percentage_error is not None])
        
        baseline = self.baseline_metrics[energy_type]
        
        # Calculate drift score (normalized degradation)
        mae_drift = (current_mae - baseline['mae']) / (baseline['mae'] + 1e-6)
        mape_drift = (current_mape - baseline['mape']) / (baseline['mape'] + 1e-6)
        
        drift_score = max(0, (mae_drift + mape_drift) / 2)
        drift_score = min(1.0, drift_score)  # Cap at 1.0
        
        # Update Prometheus metric
        model_drift_score.labels(energy_type=energy_type).set(drift_score)
        
        # Drift detected if performance degrades by >20%
        drift_detected = drift_score > 0.20
        
        result = {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'current_mae': float(current_mae),
            'baseline_mae': float(baseline['mae']),
            'current_mape': float(current_mape),
            'baseline_mape': float(baseline['mape']),
            'degradation_pct': float(drift_score * 100)
        }
        
        if drift_detected:
            logger.warning(f"Model drift detected for {energy_type}: {result}")
        
        return result
    
    def get_summary_metrics(self, energy_type: str) -> Dict:
        """Get summary metrics for an energy type"""
        if energy_type not in self.predictions:
            return {}
        
        records = self.predictions[energy_type]
        recent_with_actuals = [r for r in records[-50:] if r.actual_value is not None]
        
        if not recent_with_actuals:
            return {
                'total_predictions': len(records),
                'predictions_with_actuals': 0
            }
        
        return {
            'total_predictions': len(records),
            'predictions_with_actuals': len(recent_with_actuals),
            'recent_mae': float(np.mean([r.error for r in recent_with_actuals])),
            'recent_mape': float(np.mean([r.percentage_error for r in recent_with_actuals 
                                         if r.percentage_error is not None])),
            'latest_prediction': recent_with_actuals[-1].timestamp.isoformat(),
            'model_name': records[-1].model_name
        }
    
    def check_data_quality(self, data: pd.DataFrame, energy_type: str) -> Dict:
        """Check data quality and update metrics"""
        missing_pct = data['value'].isna().sum() / len(data) * 100
        zero_pct = (data['value'] == 0).sum() / len(data) * 100
        
        # Calculate quality score (0-1)
        quality_score = 1.0 - (missing_pct + zero_pct) / 200
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Update Prometheus metric
        data_quality_score.labels(energy_type=energy_type).set(quality_score)
        
        quality_result = {
            'quality_score': float(quality_score),
            'missing_percentage': float(missing_pct),
            'zero_percentage': float(zero_pct),
            'total_points': len(data),
            'is_healthy': quality_score > 0.8
        }
        
        if not quality_result['is_healthy']:
            logger.warning(f"Data quality issue for {energy_type}: {quality_result}")
        
        return quality_result
    
    def generate_alert(self, alert_type: str, energy_type: str, details: Dict) -> Dict:
        """Generate an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'energy_type': energy_type,
            'severity': self._determine_severity(alert_type, details),
            'details': details
        }
        
        logger.warning(f"ALERT: {alert}")
        return alert
    
    def _determine_severity(self, alert_type: str, details: Dict) -> str:
        """Determine alert severity"""
        if alert_type == 'model_drift':
            score = details.get('drift_score', 0)
            if score > 0.5:
                return 'critical'
            elif score > 0.3:
                return 'warning'
            else:
                return 'info'
        
        elif alert_type == 'data_quality':
            score = details.get('quality_score', 1.0)
            if score < 0.5:
                return 'critical'
            elif score < 0.8:
                return 'warning'
            else:
                return 'info'
        
        return 'info'


# Global monitor instance
_monitor = ModelMonitor()


def get_monitor() -> ModelMonitor:
    """Get the global monitor instance"""
    return _monitor


def export_metrics() -> bytes:
    """Export Prometheus metrics"""
    return generate_latest(REGISTRY)


if __name__ == "__main__":
    # Test the monitor
    print("Testing Model Monitor...")
    
    monitor = ModelMonitor()
    
    # Set baseline
    monitor.set_baseline_metrics('solar', {
        'mae': 250.0,
        'mape': 3.2,
        'r2': 0.98
    })
    
    # Simulate predictions
    for i in range(50):
        record = PredictionRecord(
            timestamp=datetime.now(),
            energy_type='solar',
            actual_value=5000 + np.random.normal(0, 500),
            predicted_value=5000 + np.random.normal(0, 600),
            horizon_hours=24,
            model_name='XGBoost'
        )
        monitor.record_prediction(record)
    
    # Check drift
    drift = monitor.detect_drift('solar')
    print(f"\nDrift Detection: {drift}")
    
    # Get summary
    summary = monitor.get_summary_metrics('solar')
    print(f"\nSummary Metrics: {summary}")
