# Streamlit app starts here
import streamlit as st

from w2_extract import extract_zip, load_fields

# Title of the app
st.title("W2 Extractor")

with st.sidebar:
    st.header("Field Definitions")

    uploaded_field = st.file_uploader("Upload a field definitions file", type=["json", "yaml", "yml"])

    if uploaded_field is not None:
        processed_fields = load_fields(uploaded_field)
            
        # Store in session state
        st.session_state.fields = processed_fields

        # Display field information
        st.success("Custom field definitions loaded successfully!")
        
        # Show field summary
        st.subheader("Field Summary")
        for key, field in processed_fields.items():
            st.write(f"Key: {key}")
            for k, v in field.items():
                st.write(f"{k}: {v}")
            st.write("---")
        
                    

# # File uploader widget to upload ZIP file
st.header("Document Upload")

if 'fields' in st.session_state:
    uploaded_file = st.file_uploader("Upload a zipped file", type=["zip"])

    if uploaded_file is not None:
        # Check if the file has already been processed in session state
        if st.session_state.get("uploaded_file") != uploaded_file:
            
            # Store the processed file path and filename in session state
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state['csv'] = b''
            
            csv_data = None
            with st.spinner("Processing"):
                csv_data = extract_zip(uploaded_file, fields=st.session_state.fields)
                
            st.session_state["csv"] = csv_data

    # Create a download button
    st.download_button(
        label="Download Processed CSV",
        data=st.session_state.get("csv", b''),
        file_name="processed_data.csv",
        mime="text/csv",
        disabled=(st.session_state.get("csv") is None)
    )
else:
    st.warning("Please upload field definitions in the sidebar to begin processing documents.")
        
