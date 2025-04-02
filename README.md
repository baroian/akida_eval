# Document Evaluation Web App

A Streamlit-based web application for collaborative document evaluation.

## Overview

This web application allows multiple users to evaluate documents from a dataset. Each user can classify documents by their type and subject, and add notes. The app ensures that documents are assigned uniquely to users to prevent duplicate evaluations.

## Features

- Simple login with name entry (no authentication required)
- Automatic assignment of documents to users
- Display of AI model predictions (from Gemini, DeepSeek, and ChatGPT)
- Document type and subject classification
- Notes field for additional comments
- Progress tracking
- View all evaluations in a single table
- Export evaluations as CSV

## Quick Start

### Windows Users

Double-click the `run.bat` file to start the application.

For the simplified version, you can run `run.bat --simple` from the command line.

### All Users

The easiest way to run the application is to use the included run script:

```bash
cd pk-backend/jupyter_notebook/web_app
python run.py
```

For the simplified version without document assignment:

```bash
python run.py --simple
```

The script will automatically check if all requirements are installed and if the data file exists.

## Data File Location

The application will look for the data file (`df_with_types.csv`) in either:
1. The current web app directory (`pk-backend/jupyter_notebook/web_app/`)
2. The parent directory (`pk-backend/jupyter_notebook/`)

You can place the file in either location, and the app will automatically find it.

## Manual Setup and Installation

1. Make sure you have Python 3.7+ installed

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure the CSV data file (df_with_types.csv) is in either the current directory or the parent directory

4. Run the application:

```bash
cd pk-backend/jupyter_notebook/web_app
streamlit run web_app.py
```

## Usage

1. When you first open the app, enter your name to begin evaluating documents

2. You'll be assigned 100 documents to evaluate (documents that haven't been evaluated by others)

3. For each document:
   - Read the document text
   - Review the AI model predictions
   - Select a document type (or choose "Other" to select from a dropdown)
   - Select a subject (or choose "Other" to select from a dropdown)
   - Add any notes
   - Click "Save Evaluation" to save your evaluation and move to the next document

4. Use the navigation buttons to move between documents

5. Use the sidebar to switch between document evaluation and viewing all evaluations

## Simplified Version

A simplified version of the app (`simple_app.py`) is also available for cases where you don't need the document assignment feature. This version:

- Doesn't require document assignment to users
- Allows evaluating any document in the dataset
- Still saves evaluations to a separate CSV file
- Is simpler to set up and use for small teams

To run the simplified version:

```bash
cd pk-backend/jupyter_notebook/web_app
streamlit run simple_app.py
```

Or use the run script with the `--simple` flag:

```bash
python run.py --simple
```

## Data Files

The app uses the following files:

- `df_with_types.csv`: The main dataset containing documents to evaluate
- `evaluations.csv`: Created automatically to store all evaluations
- `user_assignments.csv`: Created automatically to track which documents are assigned to which users
- `evaluations_simple.csv`: Created by the simplified app to store evaluations (if using simple_app.py)

## How It Works

1. When a user logs in, the app assigns them up to 100 documents that haven't been evaluated yet
2. The app ensures no document is assigned to multiple users at the same time
3. As users evaluate documents, their evaluations are saved to a CSV file
4. The app shows model predictions to assist in classification

## Viewing Results

Switch to the "View All Evaluations" tab to see all completed evaluations. You can also download the results as a CSV file. 