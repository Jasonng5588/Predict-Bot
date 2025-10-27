from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List
import json
import math

class DailyPredictionCombinerInput(BaseModel):
    """Input schema for Daily Prediction Combiner Tool."""
    lstm_predictions: str = Field(..., description="JSON string containing LSTM predictions with confidence scores")
    xgboost_predictions: str = Field(..., description="JSON string containing XGBoost predictions with confidence scores")
    current_price: float = Field(..., description="Current market price for reference")
    symbol: str = Field(default="", description="Trading symbol (e.g., 'BTC/USD')")

class DailyPredictionCombinerTool(BaseTool):
    """Tool for combining multiple ML model predictions using ensemble methods and generating consensus trading signals."""

    name: str = "daily_prediction_combiner_tool"
    description: str = (
        "Combines LSTM and XGBoost predictions using weighted averaging based on confidence scores. "
        "Generates consensus predictions with overall confidence levels, calculates prediction agreement, "
        "provides trading recommendations (BUY/SELL/HOLD), and formats output for Telegram with emojis."
    )
    args_schema: Type[BaseModel] = DailyPredictionCombinerInput

    def _run(self, lstm_predictions: str, xgboost_predictions: str, current_price: float, symbol: str = "") -> str:
        try:
            # Parse JSON inputs
            lstm_data = json.loads(lstm_predictions)
            xgboost_data = json.loads(xgboost_predictions)
            
            # Extract predictions and confidence scores
            lstm_prediction = float(lstm_data.get('prediction', 0))
            lstm_confidence = float(lstm_data.get('confidence', 0.5))
            
            xgboost_prediction = float(xgboost_data.get('prediction', 0))
            xgboost_confidence = float(xgboost_data.get('confidence', 0.5))
            
            # Calculate weighted ensemble prediction
            total_confidence = lstm_confidence + xgboost_confidence
            if total_confidence > 0:
                weighted_prediction = (
                    (lstm_prediction * lstm_confidence + xgboost_prediction * xgboost_confidence) 
                    / total_confidence
                )
            else:
                weighted_prediction = (lstm_prediction + xgboost_prediction) / 2
            
            # Calculate prediction agreement
            prediction_diff = abs(lstm_prediction - xgboost_prediction)
            relative_diff = prediction_diff / current_price if current_price > 0 else prediction_diff
            agreement_level = max(0, 1 - (relative_diff / 0.1))  # Agreement decreases as diff increases
            
            # Calculate overall confidence
            base_confidence = (lstm_confidence + xgboost_confidence) / 2
            consensus_confidence = base_confidence * agreement_level
            
            # Apply conservative adjustments for high disagreement
            if agreement_level < 0.5:
                consensus_confidence *= 0.7  # Reduce confidence for high disagreement
            
            # Calculate expected returns
            expected_return = (weighted_prediction - current_price) / current_price if current_price > 0 else 0
            
            # Generate trading signals
            signal = self._generate_trading_signal(expected_return, consensus_confidence)
            risk_level = self._calculate_risk_level(relative_diff, consensus_confidence)
            position_size = self._calculate_position_size(consensus_confidence, risk_level)
            
            # Format output for Telegram
            formatted_output = self._format_telegram_output(
                symbol, current_price, weighted_prediction, expected_return,
                consensus_confidence, agreement_level, signal, risk_level,
                position_size, lstm_data, xgboost_data
            )
            
            return formatted_output
            
        except json.JSONDecodeError as e:
            return f"❌ Error parsing JSON input: {str(e)}"
        except Exception as e:
            return f"❌ Error processing predictions: {str(e)}"
    
    def _generate_trading_signal(self, expected_return: float, confidence: float) -> str:
        """Generate trading signal based on expected return and confidence."""
        confidence_threshold = 0.6
        return_threshold = 0.02  # 2% minimum expected return
        
        if confidence < confidence_threshold:
            return "HOLD"
        elif expected_return > return_threshold:
            return "BUY"
        elif expected_return < -return_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_risk_level(self, relative_diff: float, confidence: float) -> str:
        """Calculate risk level based on model disagreement and confidence."""
        if relative_diff > 0.05 or confidence < 0.4:
            return "HIGH"
        elif relative_diff > 0.02 or confidence < 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_position_size(self, confidence: float, risk_level: str) -> str:
        """Calculate position size recommendation based on confidence and risk."""
        if risk_level == "HIGH":
            return "Small (1-2%)"
        elif risk_level == "MEDIUM":
            if confidence > 0.7:
                return "Medium (3-5%)"
            else:
                return "Small (1-3%)"
        else:  # LOW risk
            if confidence > 0.8:
                return "Large (5-8%)"
            else:
                return "Medium (3-5%)"
    
    def _format_telegram_output(self, symbol: str, current_price: float, prediction: float,
                              expected_return: float, confidence: float, agreement: float,
                              signal: str, risk_level: str, position_size: str,
                              lstm_data: dict, xgboost_data: dict) -> str:
        """Format output for Telegram with emojis and clear structure."""
        
        # Signal emojis
        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        
        # Format percentage values
        return_pct = expected_return * 100
        confidence_pct = confidence * 100
        agreement_pct = agreement * 100
        
        output = f"""📊 **Daily Prediction Analysis** 
{'=' * 35}

🎯 **Symbol**: {symbol if symbol else 'N/A'}
💰 **Current Price**: ${current_price:.4f}
🔮 **Predicted Price**: ${prediction:.4f}
📈 **Expected Return**: {return_pct:+.2f}%

🤖 **Model Predictions**:
├─ LSTM: ${float(lstm_data.get('prediction', 0)):.4f} (Conf: {float(lstm_data.get('confidence', 0)) * 100:.1f}%)
└─ XGBoost: ${float(xgboost_data.get('prediction', 0)):.4f} (Conf: {float(xgboost_data.get('confidence', 0)) * 100:.1f}%)

🎯 **Consensus Metrics**:
├─ Overall Confidence: {confidence_pct:.1f}%
└─ Model Agreement: {agreement_pct:.1f}%

📊 **Trading Recommendation**:
├─ Signal: {signal_emoji.get(signal, '⚪')} **{signal}**
├─ Risk Level: {risk_emoji.get(risk_level, '⚪')} {risk_level}
└─ Position Size: {position_size}

⚠️ **Risk Assessment**:
• Confidence Level: {'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'}
• Model Agreement: {'High' if agreement > 0.8 else 'Medium' if agreement > 0.6 else 'Low'}
• Recommendation: {'Strong' if confidence > 0.7 and agreement > 0.8 else 'Moderate' if confidence > 0.5 else 'Weak'}

⏰ Generated: {self._get_timestamp()}
"""
        return output
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for the report."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")