from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List
import json
import math

class LSTMStockPredictionInput(BaseModel):
    """Input schema for LSTM Stock Prediction Tool."""
    historical_data: str = Field(
        description="JSON string containing historical OHLCV stock data with keys: 'open', 'high', 'low', 'close', 'volume'. Each should be a list of numbers."
    )
    prediction_days: int = Field(
        default=1,
        description="Number of days to predict (1-7)"
    )
    sequence_length: int = Field(
        default=60,
        description="Number of historical days to use for prediction"
    )

class LSTMStockPredictionTool(BaseTool):
    """Tool for simplified LSTM-like stock price prediction using mathematical approximations."""

    name: str = "LSTM Stock Prediction Tool"
    description: str = (
        "Predicts future stock prices using simplified mathematical methods that approximate LSTM behavior. "
        "Uses moving averages, exponential smoothing, and statistical calculations for forecasting. "
        "Note: This is a simplified mathematical approximation, not actual machine learning."
    )
    args_schema: Type[BaseModel] = LSTMStockPredictionInput

    def _normalize_data(self, data: List[float]) -> tuple:
        """Min-max normalization using mathematical formulas."""
        if not data:
            return [], 0, 1
        
        min_val = min(data)
        max_val = max(data)
        
        # Avoid division by zero
        if max_val == min_val:
            return [0.5] * len(data), min_val, max_val
        
        normalized = [(x - min_val) / (max_val - min_val) for x in data]
        return normalized, min_val, max_val

    def _denormalize_data(self, normalized_data: List[float], min_val: float, max_val: float) -> List[float]:
        """Reverse min-max normalization."""
        if max_val == min_val:
            return [min_val] * len(normalized_data)
        
        return [x * (max_val - min_val) + min_val for x in normalized_data]

    def _exponential_smoothing(self, data: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing to data."""
        if not data:
            return []
        
        smoothed = [data[0]]
        for i in range(1, len(data)):
            smoothed_val = alpha * data[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_val)
        
        return smoothed

    def _calculate_moving_averages(self, data: List[float], windows: List[int]) -> Dict[int, List[float]]:
        """Calculate multiple moving averages."""
        mas = {}
        for window in windows:
            if len(data) < window:
                continue
            
            ma = []
            for i in range(len(data)):
                if i < window - 1:
                    ma.append(sum(data[:i+1]) / (i+1))
                else:
                    ma.append(sum(data[i-window+1:i+1]) / window)
            mas[window] = ma
        
        return mas

    def _calculate_momentum(self, data: List[float], period: int = 10) -> List[float]:
        """Calculate price momentum."""
        momentum = []
        for i in range(len(data)):
            if i < period:
                momentum.append(0)
            else:
                mom = data[i] - data[i-period]
                momentum.append(mom)
        return momentum

    def _lstm_approximation(self, sequences: List[List[float]]) -> List[float]:
        """
        Simplified LSTM-like approximation using mathematical operations.
        Mimics forget gate, input gate, and output gate behaviors with mathematical functions.
        """
        if not sequences:
            return []
        
        predictions = []
        
        for seq in sequences:
            if len(seq) < 3:
                predictions.append(seq[-1] if seq else 0)
                continue
            
            # Simulate LSTM gates using mathematical functions
            # Forget gate: sigmoid-like function using tanh
            recent_trend = sum(seq[-5:]) / 5 if len(seq) >= 5 else sum(seq) / len(seq)
            forget_factor = (math.tanh(recent_trend) + 1) / 2
            
            # Input gate: based on volatility and trend
            volatility = self._calculate_volatility(seq[-10:] if len(seq) >= 10 else seq)
            input_factor = 1 / (1 + volatility)
            
            # Cell state: combination of trend and momentum
            short_ma = sum(seq[-5:]) / 5 if len(seq) >= 5 else sum(seq) / len(seq)
            long_ma = sum(seq[-20:]) / 20 if len(seq) >= 20 else sum(seq) / len(seq)
            trend_factor = short_ma - long_ma
            
            # Output gate: sigmoid-like function
            output_factor = (math.tanh(trend_factor * 2) + 1) / 2
            
            # Combine factors to make prediction
            base_prediction = seq[-1]
            trend_adjustment = trend_factor * input_factor
            momentum_adjustment = (seq[-1] - seq[-2]) * forget_factor if len(seq) >= 2 else 0
            
            prediction = base_prediction + trend_adjustment * output_factor + momentum_adjustment * 0.1
            predictions.append(prediction)
        
        return predictions

    def _calculate_volatility(self, data: List[float]) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(data) < 2:
            return 0
        
        returns = [(data[i] - data[i-1]) / data[i-1] for i in range(1, len(data)) if data[i-1] != 0]
        
        if not returns:
            return 0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)

    def _calculate_metrics(self, actual: List[float], predicted: List[float]) -> Dict[str, float]:
        """Calculate RMSE and MAE metrics."""
        if len(actual) != len(predicted) or not actual:
            return {"rmse": 0, "mae": 0}
        
        # RMSE
        squared_errors = [(a - p) ** 2 for a, p in zip(actual, predicted)]
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        
        # MAE
        absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]
        mae = sum(absolute_errors) / len(absolute_errors)
        
        return {"rmse": rmse, "mae": mae}

    def _calculate_confidence(self, historical_volatility: float, prediction_days: int) -> float:
        """Calculate confidence score based on volatility and prediction horizon."""
        base_confidence = 0.8
        volatility_penalty = min(historical_volatility * 10, 0.4)
        time_penalty = min(prediction_days * 0.05, 0.3)
        
        confidence = max(base_confidence - volatility_penalty - time_penalty, 0.1)
        return round(confidence, 3)

    def _run(self, historical_data: str, prediction_days: int = 1, sequence_length: int = 60) -> str:
        try:
            # Validate prediction days
            if prediction_days < 1 or prediction_days > 7:
                return "Error: prediction_days must be between 1 and 7"
            
            # Parse historical data
            try:
                data_dict = json.loads(historical_data)
                required_keys = ['open', 'high', 'low', 'close', 'volume']
                
                if not all(key in data_dict for key in required_keys):
                    return f"Error: Historical data must contain keys: {required_keys}"
                
                close_prices = data_dict['close']
                high_prices = data_dict['high']
                low_prices = data_dict['low']
                volumes = data_dict['volume']
                
                if len(close_prices) < sequence_length:
                    return f"Error: Need at least {sequence_length} historical data points"
                
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for historical data"
            
            # Normalize data
            normalized_close, close_min, close_max = self._normalize_data(close_prices)
            normalized_high, high_min, high_max = self._normalize_data(high_prices)
            normalized_low, low_min, low_max = self._normalize_data(low_prices)
            
            # Apply exponential smoothing
            smoothed_close = self._exponential_smoothing(normalized_close)
            
            # Calculate technical indicators
            ma_dict = self._calculate_moving_averages(smoothed_close, [5, 10, 20])
            momentum = self._calculate_momentum(smoothed_close)
            
            # Create sequences for prediction
            sequences = []
            for i in range(len(smoothed_close) - sequence_length + 1):
                seq = smoothed_close[i:i + sequence_length]
                sequences.append(seq)
            
            if not sequences:
                return "Error: Not enough data to create prediction sequences"
            
            # Make predictions using LSTM approximation
            base_predictions = self._lstm_approximation([sequences[-1]])
            
            if not base_predictions:
                return "Error: Failed to generate base prediction"
            
            # Generate multi-day predictions
            predictions_normalized = []
            current_seq = sequences[-1].copy()
            
            for day in range(prediction_days):
                # Predict next value
                next_pred = self._lstm_approximation([current_seq])[0]
                predictions_normalized.append(next_pred)
                
                # Update sequence for next prediction
                current_seq = current_seq[1:] + [next_pred]
            
            # Denormalize predictions
            predictions = self._denormalize_data(predictions_normalized, close_min, close_max)
            
            # Calculate confidence and metrics
            historical_volatility = self._calculate_volatility(close_prices[-30:])
            confidence = self._calculate_confidence(historical_volatility, prediction_days)
            
            # Calculate confidence intervals (±2 standard deviations)
            recent_volatility = self._calculate_volatility(close_prices[-20:])
            confidence_range = recent_volatility * close_prices[-1] * 2
            
            # Prepare results
            results = {
                "predictions": [round(p, 2) for p in predictions],
                "confidence_score": confidence,
                "confidence_intervals": {
                    "lower": [round(p - confidence_range, 2) for p in predictions],
                    "upper": [round(p + confidence_range, 2) for p in predictions]
                },
                "model_metrics": {
                    "historical_volatility": round(historical_volatility, 4),
                    "sequence_length": sequence_length,
                    "prediction_days": prediction_days
                },
                "technical_indicators": {
                    "current_price": close_prices[-1],
                    "5_day_ma": round(sum(close_prices[-5:]) / 5, 2) if len(close_prices) >= 5 else close_prices[-1],
                    "20_day_ma": round(sum(close_prices[-20:]) / 20, 2) if len(close_prices) >= 20 else close_prices[-1]
                },
                "note": "This is a simplified mathematical approximation of LSTM behavior using statistical methods."
            }
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return f"Error in LSTM stock prediction: {str(e)}"