from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any
import requests
import json
import os
from time import sleep

class AlphaVantageDataInput(BaseModel):
    """Input schema for Alpha Vantage Data Tool."""
    stock_symbol: str = Field(
        ...,
        description="Stock symbol to fetch data for (e.g., 'AAPL', 'GOOGL', 'SPY')",
        min_length=1,
        max_length=10
    )

class AlphaVantageDataTool(BaseTool):
    """Tool for fetching real-time stock market data from Alpha Vantage API.
    
    This tool retrieves current stock quotes, price changes, volume data, and 
    basic company information for any valid stock symbol using the Alpha Vantage API.
    """

    name: str = "alpha_vantage_data_tool"
    description: str = (
        "Fetches comprehensive stock market data from Alpha Vantage API including "
        "current price, change, percentage change, volume, market cap, and company overview. "
        "Supports all major stock symbols and provides structured financial data for analysis."
    )
    args_schema: Type[BaseModel] = AlphaVantageDataInput

    def _run(self, stock_symbol: str) -> str:
        """
        Fetch stock data from Alpha Vantage API.
        
        Args:
            stock_symbol: Stock symbol to fetch data for
            
        Returns:
            JSON string containing stock data or error information
        """
        try:
            # Get API key from environment variable, fallback to demo
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
            
            # Clean up the stock symbol
            symbol = stock_symbol.upper().strip()
            
            # Base URL for Alpha Vantage API
            base_url = 'https://www.alphavantage.co/query'
            
            # Initialize result structure
            result = {
                "symbol": symbol,
                "current_price": None,
                "change": None,
                "change_percent": None,
                "volume": None,
                "market_cap": None,
                "status": "error",
                "error_message": None
            }
            
            # Set up request headers and timeout
            headers = {
                'User-Agent': 'CrewAI-AlphaVantage-Tool/1.0'
            }
            timeout = 10
            
            # First, get the global quote data
            quote_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': api_key
            }
            
            try:
                quote_response = requests.get(
                    base_url, 
                    params=quote_params, 
                    headers=headers, 
                    timeout=timeout
                )
                quote_response.raise_for_status()
                quote_data = quote_response.json()
                
                # Check for API errors
                if 'Error Message' in quote_data:
                    result["error_message"] = f"Invalid symbol: {quote_data['Error Message']}"
                    return json.dumps(result, indent=2)
                
                if 'Note' in quote_data:
                    result["error_message"] = "API rate limit exceeded. Please try again later (max 5 calls per minute)"
                    return json.dumps(result, indent=2)
                
                # Extract quote data
                if 'Global Quote' in quote_data and quote_data['Global Quote']:
                    quote = quote_data['Global Quote']
                    
                    # Parse the quote data (Alpha Vantage uses numbered keys)
                    current_price = quote.get('05. price', '0')
                    change = quote.get('09. change', '0')
                    change_percent = quote.get('10. change percent', '0%')
                    volume = quote.get('06. volume', '0')
                    
                    # Clean and convert data
                    try:
                        result["current_price"] = float(current_price)
                        result["change"] = float(change)
                        result["change_percent"] = change_percent.replace('%', '') if change_percent else '0'
                        result["volume"] = int(volume) if volume and volume != '0' else None
                    except (ValueError, TypeError) as e:
                        result["error_message"] = f"Error parsing quote data: {str(e)}"
                        return json.dumps(result, indent=2)
                
                # Add small delay to avoid rate limiting
                sleep(0.5)
                
                # Try to get company overview for market cap
                overview_params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol,
                    'apikey': api_key
                }
                
                try:
                    overview_response = requests.get(
                        base_url, 
                        params=overview_params, 
                        headers=headers, 
                        timeout=timeout
                    )
                    overview_response.raise_for_status()
                    overview_data = overview_response.json()
                    
                    # Extract market cap if available
                    if 'MarketCapitalization' in overview_data:
                        market_cap = overview_data['MarketCapitalization']
                        if market_cap and market_cap != 'None':
                            try:
                                result["market_cap"] = int(market_cap)
                            except (ValueError, TypeError):
                                pass  # Keep market_cap as None
                                
                except requests.RequestException:
                    # Overview data is optional, continue without it
                    pass
                
                # Set success status if we have basic quote data
                if result["current_price"] is not None:
                    result["status"] = "success"
                else:
                    result["error_message"] = "No quote data available for this symbol"
                    
            except requests.Timeout:
                result["error_message"] = "Request timeout - Alpha Vantage API took too long to respond"
            except requests.ConnectionError:
                result["error_message"] = "Connection error - Unable to reach Alpha Vantage API"
            except requests.HTTPError as e:
                result["error_message"] = f"HTTP error: {e.response.status_code}"
            except json.JSONDecodeError:
                result["error_message"] = "Invalid JSON response from Alpha Vantage API"
            except Exception as e:
                result["error_message"] = f"Unexpected error: {str(e)}"
                
        except Exception as e:
            result = {
                "symbol": stock_symbol.upper() if stock_symbol else "UNKNOWN",
                "current_price": None,
                "change": None,
                "change_percent": None,
                "volume": None,
                "market_cap": None,
                "status": "error",
                "error_message": f"Tool execution error: {str(e)}"
            }
        
        return json.dumps(result, indent=2)