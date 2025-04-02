#!/usr/bin/env python3
"""
Run script for Document Evaluation Tool
This script helps users to run the document evaluation web app
"""

import os
import sys
import subprocess
import argparse

def check_requirements():
    """Check if requirements are installed and install if not"""
    try:
        import streamlit
        import pandas
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Required packages installed.")

def check_data_file():
    """Check if the data file exists in any of the possible directories"""
    # Possible locations to check
    possible_locations = [
        "df_with_types.csv",                        # Current directory
        os.path.join("..", "df_with_types.csv"),    # Parent directory
        os.path.join("..", "..", "df_with_types.csv"),  # Two levels up
        os.path.join("..", "..", "..", "df_with_types.csv"),  # Three levels up
        # Absolute path provided by user
        r"C:\Users\baroi\OneDrive\Desktop\AKIDA work\pk-backend\jupyter_notebook\web_app\df_with_types.csv"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"Data file found at: {location}")
            # If file is not in the web app directory, create a copy or symlink
            if location != "df_with_types.csv":
                try:
                    # Try to copy the file to the current directory for easier access
                    import shutil
                    shutil.copy(location, "df_with_types.csv")
                    print("Created a local copy of the data file for easier access.")
                except Exception as e:
                    print(f"Note: Could not create local copy: {e}")
            return True
    
    print(f"Error: Data file 'df_with_types.csv' not found.")
    print("Please make sure the CSV file is in one of these locations:")
    for loc in possible_locations:
        print(f"  - {loc}")
    return False

def run_app(app_type="full"):
    """Run the Streamlit app"""
    app_file = "web_app.py" if app_type == "full" else "simple_app.py"
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, app_file)
    
    if not os.path.exists(app_path):
        print(f"Error: App file '{app_file}' not found at {app_path}.")
        # List all files in the current directory for debugging
        print("Files in current directory:")
        for f in os.listdir(current_dir):
            print(f"  - {f}")
        return
    
    print(f"Starting {app_type} version of the Document Evaluation Tool...")
    subprocess.call(["streamlit", "run", app_path])

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Document Evaluation Tool")
    parser.add_argument("--simple", action="store_true", help="Run simplified version (no document assignment)")
    args = parser.parse_args()
    
    app_type = "simple" if args.simple else "full"
    
    print("Document Evaluation Tool")
    print("========================")
    
    # Check requirements
    check_requirements()
    
    # Check data file
    if not check_data_file():
        return
    
    # Run app
    run_app(app_type)

if __name__ == "__main__":
    main() 