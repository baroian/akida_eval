import streamlit as st
import pandas as pd
import json
import os
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set page configuration
st.set_page_config(
    page_title="Simple Document Evaluation Tool",
    page_icon="📑",
    layout="wide"
)

# Constants and configuration
STREAMLIT_CLOUD = True  # Set to True when deploying to Streamlit Cloud
DATA_PATH = "./.streamlit" if STREAMLIT_CLOUD else "../"  # Use .streamlit folder in cloud
CSV_FILENAME = "df_with_types.csv"
EVALUATIONS_FILENAME = "evaluations_simple.csv"

# Create .streamlit directory if it doesn't exist (for Streamlit Cloud)
if STREAMLIT_CLOUD and not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)

# Helper functions
def sync_data_file():
    """Ensure the data file is in the correct location for Streamlit Cloud"""
    cloud_path = os.path.join(DATA_PATH, CSV_FILENAME)
    
    # Check if data file exists in the cloud directory
    if not os.path.exists(cloud_path) and os.path.exists(CSV_FILENAME):
        try:
            # Copy the file to the cloud directory
            shutil.copy2(CSV_FILENAME, cloud_path)
            st.success(f"Data file synced to {DATA_PATH}")
        except Exception as e:
            st.error(f"Error syncing data file: {e}")

def load_data(filename):
    """Load data from CSV file"""
    # For Streamlit Cloud, always try current directory first for the main data file
    if filename == CSV_FILENAME:
        try:
            return pd.read_csv(filename)
        except FileNotFoundError:
            pass
    
    # Try loading from the specified data path
    try:
        return pd.read_csv(os.path.join(DATA_PATH, filename))
    except FileNotFoundError:
        try:
            # If not found, try the current directory
            return pd.read_csv(filename)
        except FileNotFoundError:
            st.error(f"File {filename} not found in parent or current directory")
            return pd.DataFrame()

def save_data(df, filename):
    """Save data to CSV file"""
    # Make sure the directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH, exist_ok=True)
        
    # Save to the data path
    try:
        df.to_csv(os.path.join(DATA_PATH, filename), index=False)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        # If saving to DATA_PATH fails, try the current directory
        df.to_csv(filename, index=False)

def extract_text_for_analysis(text_str):
    """Extract readable text from the text field"""
    try:
        text_dict = json.loads(text_str.replace("'", '"')) if isinstance(text_str, str) else text_str
        if isinstance(text_dict, dict) and 'pages' in text_dict:
            extracted_text = []
            for page in text_dict['pages']:
                if 'text' in page:
                    extracted_text.append(page['text'])
            return "\n\n".join(extracted_text)
        return "Text not available in proper format"
    except:
        return "Error extracting text"

def get_doc_types_and_topics():
    """Get lists of document types and topics"""
    # This is a simplified version
    doc_types_dutch = [
        "Persbericht",
        "Rapport",
        "Beleidsnota",
        "Wetgeving",
        "Handleiding",
        "Begroting",
        "Notulen",
        "Brochure",
        "Formulier",
        "Brief",
        "Anders"
    ]
    

    doc_topics_dutch = [     
        "Gezondheid en Welzijn",
        "Ruimtelijke Ordening en Infrastructuur",
        "Onderwijs en Cultuur",
        "Milieu en Duurzaamheid",
        "Veiligheid en Openbare Orde",
        "Financiën en Belastingen",
        "Werkgelegenheid en Sociale Zaken",
        "Wonen en Huisvesting",
        "Economie en Ondernemen",
        "Immigratie en Integratie",
        "Anders"
    ]
    

    return doc_types_dutch, doc_topics_dutch

def initialize_evaluations():
    """Initialize or load the evaluations dataframe"""
    # Check if evaluations file exists in the DATA_PATH
    file_path = os.path.join(DATA_PATH, EVALUATIONS_FILENAME)
    
    try:
        # Try to read the file from the specified location
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        
        # If file doesn't exist, create a new DataFrame
        evaluations_df = pd.DataFrame(columns=[
            'doc_id', 'file_name',
            'doc_type_gemini', 'doc_type_deepseek', 'doc_type_chatgpt', 'selected_doc_type',
            'subject_gemini', 'subject_deepseek', 'subject_chatgpt', 'selected_subject',
            'notes', 'evaluator', 'evaluation_date'
        ])
        
        # Save the empty DataFrame to create the file
        save_data(evaluations_df, EVALUATIONS_FILENAME)
        return evaluations_df
        
    except Exception as e:
        st.error(f"Error initializing evaluations file: {e}")
        # Return an empty DataFrame as fallback
        return pd.DataFrame(columns=[
            'doc_id', 'file_name',
            'doc_type_gemini', 'doc_type_deepseek', 'doc_type_chatgpt', 'selected_doc_type',
            'subject_gemini', 'subject_deepseek', 'subject_chatgpt', 'selected_subject',
            'notes', 'evaluator', 'evaluation_date'
        ])

def save_evaluation(doc_id, doc_type, subject, notes, evaluator, doc_data):
    """Save document evaluation to the evaluations CSV"""
    # Load the most recent version of evaluations to prevent overwriting
    evaluations_df = initialize_evaluations()
    
    # Extract model predictions from doc_data
    doc_type_gemini = doc_data.get('doc_type_nl_gemini', '')
    doc_type_deepseek = doc_data.get('doc_type_nl_deepseek', '')
    doc_type_chatgpt = doc_data.get('doc_type_nl_chatgpt', '')
    
    subject_gemini = doc_data.get('subject_nl_gemini', '')
    subject_deepseek = doc_data.get('subject_nl_deepseek', '')
    subject_chatgpt = doc_data.get('subject_nl_chatgpt', '')
    
    file_name = doc_data.get('file_name', '')
    
    # Check if this document has already been evaluated
    if doc_id in evaluations_df['doc_id'].values:
        # Update existing evaluation
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'file_name'] = file_name
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'doc_type_gemini'] = doc_type_gemini
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'doc_type_deepseek'] = doc_type_deepseek
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'doc_type_chatgpt'] = doc_type_chatgpt
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'selected_doc_type'] = doc_type
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'subject_gemini'] = subject_gemini
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'subject_deepseek'] = subject_deepseek
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'subject_chatgpt'] = subject_chatgpt
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'selected_subject'] = subject
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'notes'] = notes
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'evaluator'] = evaluator
        evaluations_df.loc[evaluations_df['doc_id'] == doc_id, 'evaluation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Add new evaluation
        new_evaluation = pd.DataFrame([{
            'doc_id': doc_id,
            'file_name': file_name,
            'doc_type_gemini': doc_type_gemini,
            'doc_type_deepseek': doc_type_deepseek,
            'doc_type_chatgpt': doc_type_chatgpt,
            'selected_doc_type': doc_type,
            'subject_gemini': subject_gemini,
            'subject_deepseek': subject_deepseek,
            'subject_chatgpt': subject_chatgpt,
            'selected_subject': subject,
            'notes': notes,
            'evaluator': evaluator,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        evaluations_df = pd.concat([evaluations_df, new_evaluation], ignore_index=True)
    
    # Save updated evaluations
    try:
        # Save to file with locking to prevent race conditions
        save_data(evaluations_df, EVALUATIONS_FILENAME)
    except Exception as e:
        st.error(f"Error saving evaluation: {e}")
        return None
    
    return evaluations_df

# Main app function
def main():
    # Set up session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'doc_types' not in st.session_state:
        st.session_state.doc_types, st.session_state.doc_topics = get_doc_types_and_topics()
    if 'user_name' not in st.session_state:
        st.session_state.user_name = ""
    
    # App sidebar
    with st.sidebar:
        st.title("Document Evaluation Tool")
        user_name = st.text_input("Your Name", value=st.session_state.user_name)
        if user_name:
            st.session_state.user_name = user_name
        
        st.caption("Choose a section:")
        tabs = ["Evaluate Documents", "View All Evaluations", "Result Analysis"]
        selected_tab = st.radio("Navigation", tabs)
    
    # Load data
    df = load_data(CSV_FILENAME)
    evaluations_df = initialize_evaluations()
    
    if df.empty:
        st.error("Could not load document data. Please make sure 'df_with_types.csv' exists in the parent directory.")
        return
    
    if selected_tab == "Evaluate Documents":
        # Document selection input
        doc_indices = list(range(len(df)))
        index_selector = st.selectbox(
            "Select Document by Index",
            options=doc_indices,
            index=st.session_state.current_index,
            format_func=lambda x: f"Document {x+1} of {len(df)}: {df.iloc[x]['file_name']}"
        )
        st.session_state.current_index = index_selector
        
        # Get current document
        doc_data = df.iloc[st.session_state.current_index]
        doc_id = doc_data['doc_id']
        
        # Display document information first
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Document Information")
            st.write(f"**File Name:** {doc_data['file_name']}")
            st.write(f"**Origin:** {doc_data['origin']}")
            if 'date_published' in doc_data and pd.notna(doc_data['date_published']):
                st.write(f"**Date Published:** {doc_data['date_published']}")
        
        with col2:
            st.subheader("Model Predictions")
            
            st.write("**Document Type Predictions:**")
            for model in ['gemini', 'deepseek', 'chatgpt']:
                col = f'doc_type_nl_{model}'
                if col in doc_data and pd.notna(doc_data[col]):
                    st.write(f"- {model.capitalize()}: {doc_data[col]}")
            
            st.write("**Subject Predictions:**")
            for model in ['gemini', 'deepseek', 'chatgpt']:
                col = f'subject_nl_{model}'
                if col in doc_data and pd.notna(doc_data[col]):
                    st.write(f"- {model.capitalize()}: {doc_data[col]}")
        
        # Get existing evaluation data if it exists
        existing_evaluation = evaluations_df[evaluations_df['doc_id'] == doc_id]
        
        # Evaluation form
        st.subheader("Your Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document type selection
            st.write("**Select Document Type:**")
            
            # Create radio options with model predictions
            doc_type_options = []
            for model in ['gemini', 'deepseek', 'chatgpt']:
                col = f'doc_type_nl_{model}'
                if col in doc_data and pd.notna(doc_data[col]) and doc_data[col] not in [opt for opt in doc_type_options]:
                    doc_type_options.append(doc_data[col])
            
            # Add "Other" option
            doc_type_options.append("Other")
            
            # Get default value from existing evaluation
            default_doc_type = None
            if not existing_evaluation.empty and pd.notna(existing_evaluation['selected_doc_type'].iloc[0]):
                default_doc_type = existing_evaluation['selected_doc_type'].iloc[0]
                if default_doc_type not in doc_type_options:
                    # If the saved option isn't in the list, select "Other"
                    doc_type_selection = "Other"
                else:
                    doc_type_selection = default_doc_type
                
                # Show radio buttons with previously selected value
                doc_type_selection = st.radio(
                    "Document Type",
                    options=doc_type_options,
                    index=doc_type_options.index(doc_type_selection) if doc_type_selection in doc_type_options else 0,
                    key="doc_type_radio"
                )
            else:
                # Show radio buttons with no default selection for new documents
                doc_type_selection = st.radio(
                    "Document Type",
                    options=doc_type_options,
                    index=None,
                    key="doc_type_radio"
                )
            
            # Show dropdown if "Other" is selected
            if doc_type_selection == "Other":
                # Find index of default value in the doc types list
                default_index = 0
                if default_doc_type and default_doc_type in st.session_state.doc_types:
                    default_index = st.session_state.doc_types.index(default_doc_type)
                
                custom_doc_type = st.selectbox(
                    "Custom Document Type",
                    options=st.session_state.doc_types,
                    index=default_index,
                    key="custom_doc_type"
                )
            else:
                custom_doc_type = None
        
        with col2:
            # Subject selection
            st.write("**Select Subject:**")
            
            # Create radio options with model predictions
            subject_options = []
            for model in ['gemini', 'deepseek', 'chatgpt']:
                col = f'subject_nl_{model}'
                if col in doc_data and pd.notna(doc_data[col]) and doc_data[col] not in [opt for opt in subject_options]:
                    subject_options.append(doc_data[col])
            
            # Add "Other" option
            subject_options.append("Other")
            
            # Get default value from existing evaluation
            default_subject = None
            if not existing_evaluation.empty and pd.notna(existing_evaluation['selected_subject'].iloc[0]):
                default_subject = existing_evaluation['selected_subject'].iloc[0]
                if default_subject not in subject_options:
                    # If the saved option isn't in the list, select "Other"
                    subject_selection = "Other"
                else:
                    subject_selection = default_subject
                
                # Show radio buttons with previously selected value
                subject_selection = st.radio(
                    "Subject",
                    options=subject_options,
                    index=subject_options.index(subject_selection) if subject_selection in subject_options else 0,
                    key="subject_radio"
                )
            else:
                # Show radio buttons with no default selection for new documents
                subject_selection = st.radio(
                    "Subject",
                    options=subject_options,
                    index=None,
                    key="subject_radio"
                )
            
            # Show dropdown if "Other" is selected
            if subject_selection == "Other":
                # Find index of default value in the subjects list
                default_index = 0
                if default_subject and default_subject in st.session_state.doc_topics:
                    default_index = st.session_state.doc_topics.index(default_subject)
                
                custom_subject = st.selectbox(
                    "Custom Subject",
                    options=st.session_state.doc_topics,
                    index=default_index,
                    key="custom_subject"
                )
            else:
                custom_subject = None
        
        # Notes field
        notes_default = "" if existing_evaluation.empty else existing_evaluation['notes'].iloc[0]
        notes = st.text_area("Notes", value=notes_default, key="notes", height=100)
        
        # Save and navigation buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Determine final document type and subject
            final_doc_type = custom_doc_type if doc_type_selection == "Other" else doc_type_selection
            final_subject = custom_subject if subject_selection == "Other" else subject_selection
            
            save_button = st.button("Save Evaluation")
            if save_button:
                if not st.session_state.user_name:
                    st.error("Please enter your name before saving evaluations.")
                else:
                    # Save evaluation
                    save_evaluation(
                        doc_id=doc_id,
                        doc_type=final_doc_type,
                        subject=final_subject,
                        notes=notes,
                        evaluator=st.session_state.user_name,
                        doc_data=doc_data
                    )
                    
                    st.success("Evaluation saved!")
        
        with col2:
            back_button = st.button("← Back")
            if back_button:
                # Simply go back one document
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()
                
        with col3:
            next_button = st.button("Next →")
            if next_button:
                # Find next unevaluated document
                evaluated_docs = set(evaluations_df['doc_id'].values)
                next_index = None
                
                # Start from the next document
                for i in range(st.session_state.current_index + 1, len(df)):
                    if df.iloc[i]['doc_id'] not in evaluated_docs:
                        next_index = i
                        break
                
                # If not found, loop back to the beginning
                if next_index is None:
                    for i in range(0, st.session_state.current_index + 1):
                        if df.iloc[i]['doc_id'] not in evaluated_docs:
                            next_index = i
                            break
                
                # If still not found, just go to the next document
                if next_index is None:
                    if st.session_state.current_index < len(df) - 1:
                        next_index = st.session_state.current_index + 1
                    else:
                        next_index = 0
                
                st.session_state.current_index = next_index
                st.rerun()
        
        with col4:
            if 'url' in doc_data and pd.notna(doc_data['url']):
                st.write(f"**URL:** [{doc_data['url']}]({doc_data['url']})")
                st.button("Open URL")
        
        # Document text display (moved to the bottom)
        st.subheader("Document Text")
        document_text = extract_text_for_analysis(doc_data['text'])
        st.text_area("", document_text, height=400, key="doc_text")
    
    elif selected_tab == "View All Evaluations":
        st.title("All Document Evaluations")
        
        # Load evaluations
        evaluations_df = initialize_evaluations()
        
        if evaluations_df.empty:
            st.info("No evaluations have been completed yet.")
        else:
            # Format the column names for display
            column_config = {
                "file_name": "File Name",
                "doc_type_gemini": "Doc Type (Gemini)",
                "doc_type_deepseek": "Doc Type (DeepSeek)",
                "doc_type_chatgpt": "Doc Type (ChatGPT)",
                "selected_doc_type": "✓ Selected Doc Type",
                "subject_gemini": "Subject (Gemini)",
                "subject_deepseek": "Subject (DeepSeek)",
                "subject_chatgpt": "Subject (ChatGPT)",
                "selected_subject": "✓ Selected Subject",
                "notes": "Notes",
                "evaluator": "Evaluated By",
                "evaluation_date": "Evaluation Date"
            }
            
            # Display columns in a logical order
            display_columns = [
                'file_name', 
                'doc_type_gemini', 'doc_type_deepseek', 'doc_type_chatgpt', 'selected_doc_type',
                'subject_gemini', 'subject_deepseek', 'subject_chatgpt', 'selected_subject',
                'notes', 'evaluator', 'evaluation_date'
            ]
            
            # Show evaluation table
            st.dataframe(
                evaluations_df[display_columns],
                use_container_width=True,
                column_config=column_config
            )
            
            # Download button
            col1, col2 = st.columns([4, 1])
            
            with col1:
                csv = evaluations_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Evaluations",
                    data=csv,
                    file_name="document_evaluations_simple.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Delete button with warning
                delete_button = st.button(
                    "Delete All Evaluations", 
                    type="primary",
                    use_container_width=True,
                    key="delete_button"
                )
                
                # Apply custom CSS to make the button red
                st.markdown(
                    """
                    <style>
                    div[data-testid="stButton"] button[kind="primary"] {
                        background-color: #FF0000;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            
            # Show warning if button is clicked
            if delete_button:
                st.warning("⚠️ Warning: This will permanently delete all evaluation data. This action cannot be undone.")
                confirm_delete = st.button(
                    "Yes, Delete All Evaluations", 
                    type="primary",
                    key="confirm_delete"
                )
                
                if confirm_delete:
                    # Create empty dataframe with the same columns
                    empty_df = pd.DataFrame(columns=evaluations_df.columns)
                    
                    # Get the full file path
                    file_path = os.path.join(DATA_PATH, EVALUATIONS_FILENAME)
                    
                    # Save the empty dataframe
                    if os.path.exists(file_path):
                        empty_df.to_csv(file_path, index=False)
                    else:
                        save_data(empty_df, EVALUATIONS_FILENAME)
                    
                    st.success("All evaluations have been deleted.")
                    # Clear the dataframe in the current session
                    evaluations_df = empty_df
                    # Force page refresh
                    st.experimental_rerun()
    
    elif selected_tab == "Result Analysis":
        st.title("Result Analysis")
        
        if evaluations_df.empty:
            st.info("No evaluations have been completed yet. Evaluate some documents to see analysis.")
            return
        
        # Merge dataframes to get all model predictions along with human evaluations
        merged_df = pd.merge(
            evaluations_df, 
            df[['doc_id', 'doc_type_nl_gemini', 'doc_type_nl_deepseek', 'doc_type_nl_chatgpt', 
                'subject_nl_gemini', 'subject_nl_deepseek', 'subject_nl_chatgpt']], 
            on='doc_id', 
            how='left'
        )
        
        tab1, tab2, tab3, tab4 = st.tabs(["Model Accuracy", "Confusion Matrices", "Agreement Analysis", "Evaluator Statistics"])
        
        with tab1:
            st.subheader("Model Accuracy Analysis")
            
            # Calculate accuracy for document type
            doc_type_accuracy = {}
            for model in ['gemini', 'deepseek', 'chatgpt']:
                model_col = f'doc_type_nl_{model}'
                correct = merged_df[model_col] == merged_df['selected_doc_type']
                accuracy = correct.mean() * 100
                doc_type_accuracy[model] = accuracy
            
            # Calculate accuracy for subject
            subject_accuracy = {}
            for model in ['gemini', 'deepseek', 'chatgpt']:
                model_col = f'subject_nl_{model}'
                correct = merged_df[model_col] == merged_df['selected_subject']
                accuracy = correct.mean() * 100
                subject_accuracy[model] = accuracy
            
            # Plot document type accuracy
            st.write("### Document Type Prediction Accuracy")
            doc_type_df = pd.DataFrame({
                'Model': list(doc_type_accuracy.keys()),
                'Accuracy (%)': list(doc_type_accuracy.values())
            })
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Model', y='Accuracy (%)', data=doc_type_df, ax=ax)
            ax.set_ylim(0, 100)
            ax.set_title('Document Type Prediction Accuracy by Model')
            st.pyplot(fig)
            
            # Plot subject accuracy
            st.write("### Subject Prediction Accuracy")
            subject_df = pd.DataFrame({
                'Model': list(subject_accuracy.keys()),
                'Accuracy (%)': list(subject_accuracy.values())
            })
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Model', y='Accuracy (%)', data=subject_df, ax=ax)
            ax.set_ylim(0, 100)
            ax.set_title('Subject Prediction Accuracy by Model')
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Confusion Matrices")
            
            # Function to create and display confusion matrix
            def plot_confusion_matrix(y_true, y_pred, title):
                # Get unique labels
                labels = sorted(list(set(y_true) | set(y_pred)))
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels, yticklabels=labels, ax=ax)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                return fig
            
            # Document Type Confusion Matrices
            st.write("### Document Type Confusion Matrices")
            
            model_selection = st.selectbox(
                "Select Model for Document Type Confusion Matrix",
                options=['gemini', 'deepseek', 'chatgpt']
            )
            
            # Filter out rows with missing values for the selected model
            filtered_df = merged_df.dropna(subset=[f'doc_type_nl_{model_selection}', 'selected_doc_type'])
            
            if len(filtered_df) > 0:
                cm_fig = plot_confusion_matrix(
                    filtered_df['selected_doc_type'], 
                    filtered_df[f'doc_type_nl_{model_selection}'],
                    f'Confusion Matrix for {model_selection.capitalize()} - Document Type'
                )
                st.pyplot(cm_fig)
            else:
                st.warning(f"Not enough data to create confusion matrix for {model_selection} document type predictions.")
            
            # Subject Confusion Matrices
            st.write("### Subject Confusion Matrices")
            
            subject_model_selection = st.selectbox(
                "Select Model for Subject Confusion Matrix",
                options=['gemini', 'deepseek', 'chatgpt']
            )
            
            # Filter out rows with missing values for the selected model
            subject_filtered_df = merged_df.dropna(subset=[f'subject_nl_{subject_model_selection}', 'selected_subject'])
            
            if len(subject_filtered_df) > 0:
                subject_cm_fig = plot_confusion_matrix(
                    subject_filtered_df['selected_subject'], 
                    subject_filtered_df[f'subject_nl_{subject_model_selection}'],
                    f'Confusion Matrix for {subject_model_selection.capitalize()} - Subject'
                )
                st.pyplot(subject_cm_fig)
            else:
                st.warning(f"Not enough data to create confusion matrix for {subject_model_selection} subject predictions.")
        
        with tab3:
            st.subheader("Agreement Analysis")
            
            # Document Type Agreement
            st.write("### Document Type Prediction Agreement Between Models")
            
            # Create agreement matrix for document type
            doc_type_models = ['gemini', 'deepseek', 'chatgpt', 'human']
            doc_type_cols = ['doc_type_nl_gemini', 'doc_type_nl_deepseek', 'doc_type_nl_chatgpt', 'selected_doc_type']
            
            # Rename columns for the agreement calculation
            agreement_df = merged_df[doc_type_cols].copy()
            agreement_df.columns = doc_type_models
            
            # Calculate agreement matrix
            agreement_matrix = np.zeros((len(doc_type_models), len(doc_type_models)))
            
            for i, model1 in enumerate(doc_type_models):
                for j, model2 in enumerate(doc_type_models):
                    # Calculate agreement percentage
                    agreement = (agreement_df[model1] == agreement_df[model2]).mean() * 100
                    agreement_matrix[i, j] = agreement
            
            # Plot agreement heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                       xticklabels=doc_type_models, yticklabels=doc_type_models, ax=ax)
            ax.set_title('Document Type Agreement Between Models (%)')
            st.pyplot(fig)
            
            # Subject Agreement
            st.write("### Subject Prediction Agreement Between Models")
            
            # Create agreement matrix for subject
            subject_models = ['gemini', 'deepseek', 'chatgpt', 'human']
            subject_cols = ['subject_nl_gemini', 'subject_nl_deepseek', 'subject_nl_chatgpt', 'selected_subject']
            
            # Rename columns for the agreement calculation
            subject_agreement_df = merged_df[subject_cols].copy()
            subject_agreement_df.columns = subject_models
            
            # Calculate agreement matrix
            subject_agreement_matrix = np.zeros((len(subject_models), len(subject_models)))
            
            for i, model1 in enumerate(subject_models):
                for j, model2 in enumerate(subject_models):
                    # Calculate agreement percentage
                    agreement = (subject_agreement_df[model1] == subject_agreement_df[model2]).mean() * 100
                    subject_agreement_matrix[i, j] = agreement
            
            # Plot agreement heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(subject_agreement_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                       xticklabels=subject_models, yticklabels=subject_models, ax=ax)
            ax.set_title('Subject Agreement Between Models (%)')
            st.pyplot(fig)
        
        with tab4:
            st.subheader("Evaluator Statistics")
            
            # Count documents evaluated by each evaluator
            evaluator_counts = evaluations_df['evaluator'].value_counts().reset_index()
            evaluator_counts.columns = ['Evaluator', 'Documents Evaluated']
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Evaluator', y='Documents Evaluated', data=evaluator_counts, ax=ax)
            ax.set_title('Number of Documents Evaluated by Each Evaluator')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Display as a table
            st.write("### Evaluator Activity Table")
            st.dataframe(evaluator_counts, use_container_width=True)

# Run the app
if __name__ == "__main__":
    sync_data_file()
    main() 