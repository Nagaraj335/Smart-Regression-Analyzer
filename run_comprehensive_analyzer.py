"""
🚀 COMPREHENSIVE ML ANALYZER LAUNCHER
From CSV to Actionable Insights in Under 60 Seconds!

Author: Nagaraj Satish Navada
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 COMPREHENSIVE ML ANALYZER")
    print("=" * 50)
    print("From CSV to Actionable Insights in Under 60 Seconds!")
    print("Author: Nagaraj Satish Navada")
    print("=" * 50)
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "smart_regression_analyzer.py"
    
    # Check if the app file exists
    if not app_file.exists():
        print(f"❌ Error: {app_file} not found!")
        return
    
    # Check Python executable
    # Use conda environment python
    python_exe = r"C:\Users\ASUS\OneDrive\Documents\python\python.exe"
    print(f"🐍 Python: {python_exe}")
    print(f"📁 App: {app_file}")
    print()
    
    try:
        print("🌐 Starting Comprehensive ML Analyzer...")
        print("📝 The app will open in your web browser")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        print()
        
        # Run the Streamlit app
        cmd = [
            python_exe, 
            "-m", 
            "streamlit", 
            "run", 
            str(app_file),
            "--server.port=8510",
            "--server.headless=true"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Analyzer stopped by user")
    except Exception as e:
        print(f"❌ Error starting analyzer: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check if all required packages are installed")
        print("3. Ensure Python environment is properly configured")

if __name__ == "__main__":
    main()