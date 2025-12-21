"""
Stock Market Analysis System - Configuration
"""

# API Configuration
DEFAULT_PERIOD = "1y"  # Default data period
DEFAULT_INTERVAL = "1d"  # Default interval

# Technical Analysis Defaults
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
ADX_PERIOD = 14
CCI_PERIOD = 20
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3
PARABOLIC_SAR_AF_START = 0.02
PARABOLIC_SAR_AF_INCREMENT = 0.02
PARABOLIC_SAR_AF_MAX = 0.20

# Fibonacci Levels
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
