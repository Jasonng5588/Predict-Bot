from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Optional, List
import requests
import json
import time
from datetime import datetime, timedelta

class YahooFinanceDataRequest(BaseModel):
    """Input schema for Yahoo Finance Data Tool."""
    symbol: str = Field(description="Stock symbol to fetch data for (e.g., AAPL, TSLA, GOOGL)")
    period: str = Field(default="1y", description="Time period for historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)")
    include_historical: bool = Field(default=True, description="Whether to include historical data or just current price data")

class YahooFinanceDataTool(BaseTool):
    """Enhanced tool for fetching stock data with retry logic and fallback APIs."""

    name: str = "yahoo_finance_data_tool"
    description: str = (
        "Enhanced Yahoo Finance tool that fetches comprehensive stock data with retry logic, "
        "timeout handling, and fallback to alternative APIs. Returns current price, "
        "market statistics, historical data, and financial metrics with improved reliability."
    )
    args_schema: Type[BaseModel] = YahooFinanceDataRequest

    # Pydantic fields for configuration parameters
    max_retries: int = Field(default=3, description="Maximum number of retry attempts for failed requests")
    base_timeout: int = Field(default=10, description="Base timeout in seconds for HTTP requests")
    retry_delay: int = Field(default=1, description="Delay in seconds between retry attempts")

    def __init__(self, max_retries: int = 3, base_timeout: int = 10, retry_delay: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.retry_delay = retry_delay

    def _make_request_with_retry(self, url: str, params: Dict = None, headers: Dict = None, timeout: int = 10) -> Optional[requests.Response]:
        """Make HTTP request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=timeout + (attempt * 2)  # Increase timeout with each retry
                )
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1) * 2)  # Longer wait for rate limiting
                        continue
                elif e.response.status_code >= 500:  # Server error
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
        return None

    def _get_current_data_yahoo(self, symbol: str) -> Dict[str, Any]:
        """Fetch current stock data from Yahoo Finance with retry logic."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/quoteSummary/{symbol}"
            params = {
                "modules": "price,summaryDetail,defaultKeyStatistics,financialData"
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = self._make_request_with_retry(url, params=params, headers=headers, timeout=self.base_timeout)
            
            if not response:
                return {"error": "Failed to get response after retries"}
                
            data = response.json()
            
            if 'quoteSummary' not in data or not data['quoteSummary']['result']:
                return {"error": f"No data found for symbol {symbol}"}
            
            result = data['quoteSummary']['result'][0]
            
            # Extract relevant data
            price_data = result.get('price', {})
            summary_detail = result.get('summaryDetail', {})
            key_stats = result.get('defaultKeyStatistics', {})
            financial_data = result.get('financialData', {})
            
            current_data = {
                "symbol": symbol.upper(),
                "current_price": self._safe_get_value(price_data.get('regularMarketPrice')),
                "previous_close": self._safe_get_value(price_data.get('regularMarketPreviousClose')),
                "change": self._safe_get_value(price_data.get('regularMarketChange')),
                "change_percent": self._safe_get_value(price_data.get('regularMarketChangePercent')),
                "volume": self._safe_get_value(price_data.get('regularMarketVolume')),
                "avg_volume": self._safe_get_value(price_data.get('averageDailyVolume3Month')),
                "market_cap": self._safe_get_value(price_data.get('marketCap')),
                "day_high": self._safe_get_value(price_data.get('regularMarketDayHigh')),
                "day_low": self._safe_get_value(price_data.get('regularMarketDayLow')),
                "fifty_two_week_high": self._safe_get_value(summary_detail.get('fiftyTwoWeekHigh')),
                "fifty_two_week_low": self._safe_get_value(summary_detail.get('fiftyTwoWeekLow')),
                "pe_ratio": self._safe_get_value(key_stats.get('trailingPE')),
                "dividend_yield": self._safe_get_value(summary_detail.get('dividendYield')),
                "beta": self._safe_get_value(summary_detail.get('beta')),
                "eps": self._safe_get_value(key_stats.get('trailingEps')),
                "currency": price_data.get('currency', 'USD'),
                "exchange": price_data.get('exchangeName', 'Unknown'),
                "data_source": "yahoo_finance",
                "last_updated": datetime.now().isoformat()
            }
            
            return current_data
            
        except requests.exceptions.Timeout:
            return {"error": f"Timeout while fetching data for {symbol} after {self.max_retries} attempts"}
        except requests.exceptions.ConnectionError:
            return {"error": f"Connection error while fetching data for {symbol} after {self.max_retries} attempts"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP error {e.response.status_code} while fetching data for {symbol}"}
        except Exception as e:
            return {"error": f"Error parsing current data from Yahoo Finance: {str(e)}"}

    def _get_fallback_current_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback method using alternative API endpoints."""
        try:
            # Try Yahoo Finance query API (simpler endpoint)
            fallback_urls = [
                f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?modules=price",
                f"https://finance.yahoo.com/quote/{symbol}/history"  # This would need parsing but is more reliable
            ]
            
            for url in fallback_urls:
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    
                    response = self._make_request_with_retry(url, headers=headers, timeout=15)
                    
                    if response and url.endswith('modules=price'):
                        data = response.json()
                        if 'quoteSummary' in data and data['quoteSummary']['result']:
                            price_data = data['quoteSummary']['result'][0].get('price', {})
                            
                            return {
                                "symbol": symbol.upper(),
                                "current_price": self._safe_get_value(price_data.get('regularMarketPrice')),
                                "previous_close": self._safe_get_value(price_data.get('regularMarketPreviousClose')),
                                "change": self._safe_get_value(price_data.get('regularMarketChange')),
                                "change_percent": self._safe_get_value(price_data.get('regularMarketChangePercent')),
                                "currency": price_data.get('currency', 'USD'),
                                "data_source": "yahoo_fallback",
                                "last_updated": datetime.now().isoformat(),
                                "note": "Limited data from fallback source"
                            }
                    
                except Exception:
                    continue
            
            return {"error": f"All fallback methods failed for symbol {symbol}"}
            
        except Exception as e:
            return {"error": f"Fallback data fetch failed: {str(e)}"}

    def _get_historical_data_yahoo(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fetch historical stock data from Yahoo Finance with retry logic."""
        try:
            # Calculate timestamps
            end_time = int(time.time())
            
            # Period mapping
            period_mapping = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
            }
            
            days = period_mapping.get(period, 365)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d',
                'includePrePost': 'true',
                'events': 'div,splits'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = self._make_request_with_retry(url, params=params, headers=headers, timeout=15)
            
            if not response:
                return {"error": "Failed to get historical data response after retries"}
            
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                return {"error": f"No historical data found for symbol {symbol}"}
            
            result = data['chart']['result'][0]
            
            timestamps = result['timestamp']
            quote_data = result['indicators']['quote'][0]
            
            historical_prices = []
            for i, timestamp in enumerate(timestamps):
                if quote_data['close'][i] is not None:
                    historical_prices.append({
                        "date": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                        "open": quote_data['open'][i],
                        "high": quote_data['high'][i],
                        "low": quote_data['low'][i],
                        "close": quote_data['close'][i],
                        "volume": quote_data['volume'][i] or 0
                    })
            
            return {
                "symbol": symbol.upper(),
                "period": period,
                "historical_data": historical_prices,
                "data_points": len(historical_prices),
                "data_source": "yahoo_finance"
            }
            
        except requests.exceptions.Timeout:
            return {"error": f"Timeout while fetching historical data for {symbol} after {self.max_retries} attempts"}
        except requests.exceptions.ConnectionError:
            return {"error": f"Connection error while fetching historical data for {symbol} after {self.max_retries} attempts"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP error {e.response.status_code} while fetching historical data for {symbol}"}
        except Exception as e:
            return {"error": f"Error parsing historical data from Yahoo Finance: {str(e)}"}

    def _get_fallback_historical_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fallback method for historical data using alternative endpoints."""
        try:
            # Try alternative Yahoo Finance chart endpoint
            end_time = int(time.time())
            period_mapping = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
            }
            
            days = period_mapping.get(period, 365)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            fallback_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': start_time,
                'period2': end_time,
                'interval': '1d'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self._make_request_with_retry(fallback_url, params=params, headers=headers, timeout=20)
            
            if response:
                data = response.json()
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    timestamps = result['timestamp']
                    quote_data = result['indicators']['quote'][0]
                    
                    historical_prices = []
                    for i, timestamp in enumerate(timestamps):
                        if quote_data['close'][i] is not None:
                            historical_prices.append({
                                "date": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                                "close": quote_data['close'][i],
                                "volume": quote_data['volume'][i] or 0
                            })
                    
                    return {
                        "symbol": symbol.upper(),
                        "period": period,
                        "historical_data": historical_prices,
                        "data_points": len(historical_prices),
                        "data_source": "yahoo_fallback",
                        "note": "Limited historical data from fallback source"
                    }
            
            return {"error": f"Fallback historical data fetch failed for symbol {symbol}"}
            
        except Exception as e:
            return {"error": f"Fallback historical data fetch failed: {str(e)}"}

    def _safe_get_value(self, data_dict: Any) -> Any:
        """Safely extract raw value from Yahoo Finance data structure."""
        if isinstance(data_dict, dict) and 'raw' in data_dict:
            return data_dict['raw']
        return data_dict

    def _run(self, symbol: str, period: str = "1y", include_historical: bool = True) -> str:
        """
        Fetch Yahoo Finance data for the specified stock symbol with enhanced error handling.
        
        Args:
            symbol: Stock symbol to fetch data for
            period: Time period for historical data
            include_historical: Whether to include historical data
            
        Returns:
            JSON string containing stock data
        """
        try:
            # Rate limiting - wait between requests
            time.sleep(0.1)
            
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                return json.dumps({"error": "Invalid symbol provided"})
            
            symbol = symbol.upper().strip()
            
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "retry_attempts": self.max_retries
            }
            
            # Fetch current data with retry logic and fallback
            current_data = self._get_current_data_yahoo(symbol)
            
            if "error" in current_data:
                # Try fallback method
                fallback_data = self._get_fallback_current_data(symbol)
                if "error" not in fallback_data:
                    result["current_data"] = fallback_data
                    result["status"] = "partial_success"
                    result["note"] = "Using fallback data source"
                else:
                    return json.dumps({
                        "status": "error",
                        "error": f"Primary error: {current_data['error']}. Fallback error: {fallback_data['error']}",
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat()
                    })
            else:
                result["current_data"] = current_data
            
            # Fetch historical data if requested
            if include_historical:
                time.sleep(0.2)  # Additional rate limiting
                historical_data = self._get_historical_data_yahoo(symbol, period)
                
                if "error" in historical_data:
                    # Try fallback method for historical data
                    fallback_historical = self._get_fallback_historical_data(symbol, period)
                    if "error" not in fallback_historical:
                        result["historical_data"] = fallback_historical
                        if result["status"] == "success":
                            result["status"] = "partial_success"
                            result["note"] = "Historical data from fallback source"
                    else:
                        result["historical_data"] = {"error": historical_data["error"]}
                        result["status"] = "partial_success"
                        result["note"] = "Current data available, historical data failed"
                else:
                    result["historical_data"] = historical_data
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
                "symbol": symbol if 'symbol' in locals() else "unknown",
                "timestamp": datetime.now().isoformat(),
                "retry_attempts": self.max_retries
            }
            return json.dumps(error_result, indent=2)