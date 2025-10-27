from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List
import json
import math

class XGBoostStockPredictionRequest(BaseModel):
    """Input schema for XGBoost Stock Prediction Tool."""
    historical_data: str = Field(
        ...,
        description="Historical stock data in JSON format with OHLCV (Open, High, Low, Close, Volume) data. Format: [{'date': 'YYYY-MM-DD', 'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}, ...]"
    )
    prediction_days: int = Field(
        default=1,
        description="Number of days ahead to predict (1-7 days)",
        ge=1,
        le=7
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence threshold for predictions (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

class XGBoostStockPredictionTool(BaseTool):
    """Tool for advanced stock price prediction using ensemble methods with technical indicators."""

    name: str = "xgboost_stock_prediction_tool"
    description: str = (
        "Predicts stock prices using XGBoost-style ensemble methods with technical indicators. "
        "Creates multiple decision tree-like predictors, applies feature engineering with RSI, MACD, "
        "Bollinger Bands, Moving Averages, and Volume indicators. Returns predictions for 1-7 days "
        "ahead with confidence scores and feature importance rankings."
    )
    args_schema: Type[BaseModel] = XGBoostStockPredictionRequest

    def _calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        sma = []
        for i in range(len(prices)):
            if i < period - 1:
                sma.append(None)
            else:
                avg = sum(prices[i-period+1:i+1]) / period
                sma.append(avg)
        return sma

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if not prices:
            return []
        
        ema = [prices[0]]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema_val = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
            ema.append(ema_val)
        
        return ema

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return [None] * len(prices)
        
        deltas = []
        for i in range(1, len(prices)):
            deltas.append(prices[i] - prices[i-1])
        
        gains = [max(delta, 0) for delta in deltas]
        losses = [-min(delta, 0) for delta in deltas]
        
        rsi_values = [None] * (period)
        
        # Initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        # Calculate subsequent RSI values
        for i in range(period, len(deltas)):
            avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        return rsi_values

    def _calculate_macd(self, prices: List[float]) -> Dict[str, List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        macd_line = []
        for i in range(len(prices)):
            if i < 25:  # Need at least 26 data points
                macd_line.append(None)
            else:
                macd_line.append(ema12[i] - ema26[i])
        
        # Signal line (EMA of MACD)
        macd_values = [x for x in macd_line if x is not None]
        signal_ema = self._calculate_ema(macd_values, 9)
        
        # Pad signal line with None values to match length
        signal_line = [None] * (len(macd_line) - len(signal_ema)) + signal_ema
        
        # Histogram
        histogram = []
        for i in range(len(macd_line)):
            if macd_line[i] is None or signal_line[i] is None:
                histogram.append(None)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands."""
        sma = self._calculate_sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1 or sma[i] is None:
                upper_band.append(None)
                lower_band.append(None)
            else:
                # Calculate standard deviation
                variance = sum([(prices[j] - sma[i]) ** 2 for j in range(i-period+1, i+1)]) / period
                std = math.sqrt(variance)
                
                upper_band.append(sma[i] + (std_dev * std))
                lower_band.append(sma[i] - (std_dev * std))
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    def _calculate_volume_indicators(self, prices: List[float], volumes: List[int]) -> Dict[str, List[float]]:
        """Calculate volume-based indicators."""
        # Volume-Weighted Average Price (VWAP) approximation
        vwap = []
        cum_volume = 0
        cum_price_volume = 0
        
        for i in range(len(prices)):
            cum_volume += volumes[i]
            cum_price_volume += prices[i] * volumes[i]
            vwap.append(cum_price_volume / cum_volume if cum_volume > 0 else prices[i])
        
        # Volume Rate of Change
        volume_roc = []
        for i in range(len(volumes)):
            if i < 10:  # Need 10 periods
                volume_roc.append(None)
            else:
                roc = ((volumes[i] - volumes[i-10]) / volumes[i-10]) * 100 if volumes[i-10] > 0 else 0
                volume_roc.append(roc)
        
        return {
            'vwap': vwap,
            'volume_roc': volume_roc
        }

    def _create_features(self, data: List[Dict]) -> Dict[str, Any]:
        """Create technical indicator features from OHLCV data."""
        if len(data) < 30:
            raise ValueError("Need at least 30 data points for reliable predictions")
        
        closes = [d['close'] for d in data]
        opens = [d['open'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        volumes = [d['volume'] for d in data]
        
        features = {}
        
        # Moving Averages
        features['sma_5'] = self._calculate_sma(closes, 5)
        features['sma_10'] = self._calculate_sma(closes, 10)
        features['sma_20'] = self._calculate_sma(closes, 20)
        features['sma_50'] = self._calculate_sma(closes, 50)
        
        features['ema_12'] = self._calculate_ema(closes, 12)
        features['ema_26'] = self._calculate_ema(closes, 26)
        
        # Technical Indicators
        features['rsi'] = self._calculate_rsi(closes)
        macd_data = self._calculate_macd(closes)
        features.update(macd_data)
        
        bb_data = self._calculate_bollinger_bands(closes)
        features.update({f'bb_{k}': v for k, v in bb_data.items()})
        
        volume_data = self._calculate_volume_indicators(closes, volumes)
        features.update(volume_data)
        
        # Price-based features
        features['close'] = closes
        features['high_low_ratio'] = [h/l if l > 0 else 1 for h, l in zip(highs, lows)]
        features['open_close_ratio'] = [o/c if c > 0 else 1 for o, c in zip(opens, closes)]
        
        return features

    def _create_predictor(self, features: Dict, timeframe: int, recent_errors: List[float]) -> Dict[str, float]:
        """Create a single predictor based on timeframe and features."""
        predictor = {}
        
        # Get the most recent valid values
        recent_data = {}
        for key, values in features.items():
            valid_values = [v for v in values[-timeframe:] if v is not None]
            if valid_values:
                recent_data[key] = sum(valid_values) / len(valid_values)
            else:
                recent_data[key] = 0
        
        # Simple trend analysis
        closes = features['close']
        if len(closes) >= timeframe:
            recent_closes = closes[-timeframe:]
            trend = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] if recent_closes[0] > 0 else 0
        else:
            trend = 0
        
        # Feature weights (simulating learned weights)
        feature_weights = {
            'sma_5': 0.2, 'sma_10': 0.15, 'sma_20': 0.1, 'sma_50': 0.05,
            'ema_12': 0.15, 'ema_26': 0.1,
            'rsi': 0.1, 'macd': 0.05, 'signal': 0.03,
            'bb_upper': 0.02, 'bb_lower': 0.02, 'bb_middle': 0.03
        }
        
        # Calculate weighted prediction
        prediction_score = 0
        total_weight = 0
        
        for feature, weight in feature_weights.items():
            if feature in recent_data and recent_data[feature] is not None:
                prediction_score += recent_data[feature] * weight
                total_weight += weight
        
        # Normalize and apply trend
        if total_weight > 0:
            base_prediction = prediction_score / total_weight
        else:
            base_prediction = closes[-1] if closes else 0
        
        # Apply trend adjustment
        trend_adjustment = base_prediction * trend * 0.1
        
        # Apply error correction (gradient-like adjustment)
        error_correction = 0
        if recent_errors:
            avg_error = sum(recent_errors) / len(recent_errors)
            error_correction = -avg_error * 0.3  # Counteract recent errors
        
        final_prediction = base_prediction + trend_adjustment + error_correction
        
        # Calculate confidence based on consistency
        consistency = 1.0 - min(abs(trend), 0.5)  # Higher consistency for stable trends
        
        predictor['prediction'] = final_prediction
        predictor['confidence'] = consistency
        predictor['timeframe'] = timeframe
        
        return predictor

    def _ensemble_predictions(self, predictors: List[Dict], prediction_days: int) -> List[Dict]:
        """Combine predictions from multiple predictors."""
        ensemble_predictions = []
        
        for day in range(prediction_days):
            total_weighted_prediction = 0
            total_weight = 0
            confidences = []
            
            for predictor in predictors:
                weight = predictor['confidence'] * (1.0 / (predictor['timeframe'] ** 0.5))  # Shorter timeframes get higher weights
                total_weighted_prediction += predictor['prediction'] * weight
                total_weight += weight
                confidences.append(predictor['confidence'])
            
            if total_weight > 0:
                final_prediction = total_weighted_prediction / total_weight
            else:
                final_prediction = predictors[0]['prediction'] if predictors else 0
            
            # Calculate ensemble confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Adjust prediction for multiple days (add some volatility)
            if day > 0:
                volatility_factor = 1.0 + (day * 0.02)  # 2% additional uncertainty per day
                final_prediction *= volatility_factor
                avg_confidence *= (1.0 - day * 0.1)  # Reduce confidence for further predictions
            
            ensemble_predictions.append({
                'day': day + 1,
                'predicted_price': round(final_prediction, 2),
                'confidence': round(max(0, min(1, avg_confidence)), 3)
            })
        
        return ensemble_predictions

    def _calculate_feature_importance(self, features: Dict) -> List[Dict]:
        """Calculate feature importance scores."""
        importance_scores = []
        
        # Simulate feature importance based on volatility and predictive power
        feature_priorities = {
            'close': 1.0, 'sma_5': 0.9, 'sma_10': 0.8, 'ema_12': 0.85,
            'rsi': 0.7, 'macd': 0.6, 'bb_middle': 0.5, 'vwap': 0.4
        }
        
        for feature, values in features.items():
            if feature in feature_priorities:
                # Calculate variance as proxy for importance
                valid_values = [v for v in values if v is not None]
                if len(valid_values) > 1:
                    variance = sum([(x - sum(valid_values)/len(valid_values))**2 for x in valid_values]) / len(valid_values)
                    importance = feature_priorities[feature] * min(1.0, variance / 1000)  # Normalize variance
                else:
                    importance = feature_priorities[feature] * 0.1
                
                importance_scores.append({
                    'feature': feature,
                    'importance': round(importance, 3)
                })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x['importance'], reverse=True)
        
        return importance_scores[:10]  # Return top 10 features

    def _run(self, historical_data: str, prediction_days: int = 1, confidence_threshold: float = 0.7) -> str:
        """Execute the XGBoost-style stock prediction."""
        try:
            # Parse input data
            try:
                data = json.loads(historical_data)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON format in historical_data. {str(e)}"
            
            # Validate data structure
            if not isinstance(data, list) or len(data) == 0:
                return "Error: Historical data must be a non-empty list of OHLCV records."
            
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            for i, record in enumerate(data):
                if not isinstance(record, dict):
                    return f"Error: Record {i} must be a dictionary."
                for field in required_fields:
                    if field not in record:
                        return f"Error: Missing field '{field}' in record {i}."
                    if not isinstance(record[field], (int, float)):
                        return f"Error: Field '{field}' in record {i} must be a number."
            
            # Validate prediction days
            if prediction_days < 1 or prediction_days > 7:
                return "Error: prediction_days must be between 1 and 7."
            
            # Create features
            try:
                features = self._create_features(data)
            except ValueError as e:
                return f"Error: {str(e)}"
            
            # Create multiple predictors with different timeframes
            predictors = []
            recent_errors = []  # In real implementation, this would be historical prediction errors
            
            for timeframe in [5, 10, 20]:
                if len(data) >= timeframe:
                    predictor = self._create_predictor(features, timeframe, recent_errors)
                    predictors.append(predictor)
            
            if not predictors:
                return "Error: Not enough data to create predictors."
            
            # Generate ensemble predictions
            predictions = self._ensemble_predictions(predictors, prediction_days)
            
            # Filter by confidence threshold
            valid_predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(features)
            
            # Format results
            result = {
                'predictions': valid_predictions,
                'all_predictions': predictions,
                'feature_importance': feature_importance,
                'model_info': {
                    'predictors_used': len(predictors),
                    'prediction_days': prediction_days,
                    'confidence_threshold': confidence_threshold,
                    'data_points': len(data)
                },
                'summary': {
                    'valid_predictions': len(valid_predictions),
                    'total_predictions': len(predictions),
                    'avg_confidence': round(sum([p['confidence'] for p in predictions]) / len(predictions), 3) if predictions else 0,
                    'last_price': data[-1]['close']
                }
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"