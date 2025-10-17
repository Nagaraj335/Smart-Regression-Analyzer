"""
Vercel deployment wrapper for Smart Regression Analyzer
This file serves as the entry point for Vercel hosting
"""

import streamlit as st
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import and run the main Streamlit app
if __name__ == "__main__":
    # Import the main application
    import smart_regression_analyzer
    
    # Streamlit apps don't need explicit execution when deployed
    # The smart_regression_analyzer.py file will run automatically