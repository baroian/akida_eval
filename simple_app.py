import streamlit as st
import pandas as pd
import json
import os
import shutil
from datetime import datetime

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
        "Agenda", 
        "Besluitenlijst", 
        "Notulen", 
        "Rapport", 
        "Begroting", 
        "Jaarverslag", 
        "Brief", 
        "Persbericht",
        "Beleidsdocument",
        "Verordening",
        "Motie",
        "Amendement",
        "Aanvraag",
        "Bezwaarschrift",
        "Overeenkomst",
        "Anders"
    ]
    
    doc_topics_dutch = [
        "Bestuur en Organisatie",
        "Financiën en Economie",
        "Milieu en Duurzaamheid",
        "Ruimtelijke Ordening",
        "Verkeer en Vervoer",
        "Onderwijs en Cultuur",
        "Zorg en Welzijn",
        "Wonen en Bouwen",
        "Sport en Recreatie",
        "Openbare Orde en Veiligheid",
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
        tabs = ["Evaluate Documents", "View All Evaluations"]
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
            else:
                doc_type_selection = doc_type_options[0] if doc_type_options else None
            
            # Show radio buttons
            doc_type_selection = st.radio(
                "Document Type",
                options=doc_type_options,
                index=doc_type_options.index(doc_type_selection) if doc_type_selection in doc_type_options else 0,
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
            else:
                subject_selection = subject_options[0] if subject_options else None
            
            # Show radio buttons
            subject_selection = st.radio(
                "Subject",
                options=subject_options,
                index=subject_options.index(subject_selection) if subject_selection in subject_options else 0,
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
        
        # Save button
        col1, col2 = st.columns(2)
        
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
            next_button = st.button("Next →")
            if next_button:
                if st.session_state.current_index < len(df) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        # Document text display (moved to the bottom)
        st.subheader("Document Text")
        document_text = extract_text_for_analysis(doc_data['text'])
        st.text_area("", document_text, height=400, key="doc_text")
        
        if 'url' in doc_data and pd.notna(doc_data['url']):
            st.caption(f"URL: [{doc_data['url']}]({doc_data['url']})")
            if st.button("Open URL"):
                st.markdown(f"<a href='{doc_data['url']}' target='_blank'>Click to open document URL</a>", unsafe_allow_html=True)
    
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
            csv = evaluations_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Evaluations",
                data=csv,
                file_name="document_evaluations_simple.csv",
                mime="text/csv"
            )

# Run the app
if __name__ == "__main__":
    sync_data_file()
    main() 