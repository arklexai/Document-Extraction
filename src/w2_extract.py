from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import json
import pickle
import threading
import time
import pandas as pd

import zipfile
import tempfile
from io import BytesIO

import argparse

import yaml

from kv_extract import KVExtractDocument
from fv_extract import OptimizedExtraction
import ocr_load
from evaluation import compare_datasets

w2_fields = {
    'a': {"desc": "a Employee's Social Security Number (SSN)", "type": "string", 'example': '123456789', 'required': True},
    'e': {"desc": "e Employee's Full Name", "type": "string", 'example': 'John Smith', 'required': True},
    'f': {"desc": "f Employee’s address, and ZIP code", "type": "string", 'example': '3409 OLIVER ST UNIT A BAKERSFIELD CA 93307', 'required': True},
    '1': {"desc": "1 Wages tips other comp", "type": "float", 'example': 1234.56, 'required': True},
    '5': {"desc": "5 Medicare wages and tips", "type": "float", 'example': 1234.56, 'required': True},
    '12a': {"desc": "12a code ", "type": "string", 'example': 'A 321.80', 'required': False},
    '12b': {"desc": "12b code", "type": "string", 'example': 'CC 261.80', 'required': False},
    '14': {"desc": "14 Other", "type": "string", 'example': 'UCRP 1,374.26 DCP-REG 916.18', 'required': False}
}

_checkpoint_lock = threading.Lock()

def get_file_cache_key(filepath: str, ocr_method: str) -> str:
    """Generate a unique cache key for a file based on its absolute path."""
    # Use absolute path to ensure uniqueness across different directories
    abs_path = os.path.abspath(filepath)
    # Create a hash to keep keys manageable but unique
    return hashlib.md5(abs_path.encode()).hexdigest()[:16] + "_" + os.path.basename(filepath) + "_" + ocr_method

def get_batch_id_zip(zip_file):
    # Get file list
    file_info = []
    with zipfile.ZipFile(zip_file, 'r') as zf:
        for info in zf.infolist():
            file_info.append((info.filename, info.file_size))
    
    # Sort to ensure consistent ordering
    file_info.sort()
    
    # Create a string representation of the file list
    file_list_str = str(file_info)
    
    # Generate hash from the file list
    return hashlib.md5(file_list_str.encode()).hexdigest()

def get_batch_id_dir(dirpath: str) -> str:
    """Generate a unique batch ID for a directory based on its contents."""
    # Get list of PDF files and their sizes
    files = [f for f in os.listdir(dirpath) if f.endswith('.pdf') or f.endswith('.png') or f.endswith('.jpg')]
    file_info = []
    
    for filename in sorted(files):
        filepath = os.path.join(dirpath, filename)
        if os.path.exists(filepath):
            file_info.append((filename, os.path.getsize(filepath)))
    
    # Create a string representation and hash it
    file_list_str = str(file_info)
    return hashlib.md5(file_list_str.encode()).hexdigest()

def get_batch_file(batch_id):
    """Get the path to the batch results file."""
    return f"batch_{batch_id}.pkl"

def save_batch_results(batch_id, processed_files, results, ocr_cache=None):
    """Save batch results to file."""
    with _checkpoint_lock:
        batch_file = get_batch_file(batch_id)
        
        # Load existing data first to merge with current data
        existing_processed = {}
        existing_results = {}
        existing_ocr_cache = {}
        
        if os.path.exists(batch_file):
            try:
                with open(batch_file, 'rb') as f:
                    existing_data = pickle.load(f)
                    existing_processed = existing_data.get('processed_files', {})
                    existing_results = existing_data.get('results', {})
                    existing_ocr_cache = existing_data.get('ocr_cache', {})
            except Exception as e:
                print(f"Error loading existing data for batch {batch_id}: {e}")
                pass  # If file is corrupted, start fresh
        
        # Merge with existing data
        merged_processed = {**existing_processed, **processed_files}
        merged_results = {**existing_results, **results}
        merged_ocr_cache = existing_ocr_cache
        if ocr_cache is not None:
            merged_ocr_cache.update(ocr_cache)
        
        batch_data = {
            'processed_files': merged_processed,
            'results': merged_results,
            'ocr_cache': merged_ocr_cache
        }
        
        # Write to temporary file first, then rename for atomic operation
        temp_file = batch_file + '.tmp'
        with open(temp_file, 'wb') as f:
            pickle.dump(batch_data, f)
        os.rename(temp_file, batch_file)

def load_batch_results(batch_id):
    """Load existing batch results if available."""
    with _checkpoint_lock:
        batch_file = get_batch_file(batch_id)
        if os.path.exists(batch_file):
            try:
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    return (
                        batch_data.get('processed_files', {}), 
                        batch_data.get('results', {}), 
                        batch_data.get('ocr_cache', {})
                    )
            except Exception as e:
                print(f"Error loading existing data for batch {batch_id}: {e}")
                pass  # If file is corrupted, return empty
        return {}, {}, {}


def clear_batch_results(batch_id):
    """Clear only field extraction results, keeping OCR cache."""
    batch_file = get_batch_file(batch_id)
    if os.path.exists(batch_file):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        
        # Keep OCR cache but clear field extraction results
        batch_data['processed_files'] = {}
        batch_data['results'] = {}
        
        with open(batch_file, 'wb') as f:
            pickle.dump(batch_data, f)
        print(f"Cleared field extraction results, kept OCR cache: {batch_file}")

def group_mismatches_by_file(mismatches):
    """Group mismatches by filename."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for mismatch in mismatches:
        filename = mismatch['name']
        grouped[filename].append(mismatch)
    return dict(grouped)

def get_cached_file_data(filename, batch_id, ocr_method):
    """Get cached OCR data for a specific file."""
    _, _, ocr_cache = load_batch_results(batch_id)
    
    # Find the OCR cache entry for this file
    for cache_key, cached_data in ocr_cache.items():
        if filename in cache_key and ocr_method in cache_key:
            return cached_data
    return None

def load_fields(file) -> dict[str, dict]:
    _, ext = os.path.splitext(file.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    try:
        with open(tmp_path, 'r') as f:
            if ext.lower() in ['.yaml', '.yml']:
                custom_fields = yaml.safe_load(f)
            elif ext.lower() == '.json':
                custom_fields = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .json, .yaml, or .yml")
        
        return custom_fields
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_file(filepath: str, fields: dict[str, dict]=w2_fields, max_attempts: int=3, batch_id: str=None, filename: str=None, model_type: str="gpt-4o-mini", ocr_method: str="textract-kv", spatial: bool=False, extractor: OptimizedExtraction=None, save_batch: bool=True) -> list[dict[str, str]]:
    '''
    Extracts W2 fields from a PDF file using Textract.
    
    Parameters:
    filepath (str): The path to the PDF file.
    fields (dict): A dictionary containing field descriptions, types, and examples.
    max_attempts (int): The maximum number of attempts to extract fields.
    
    Returns:
    list: A list of dictionaries containing the extracted fields for each page.
    '''
    print(f"Processing {filepath}")

    file_cache_key =  get_file_cache_key(filepath, ocr_method)

    # Load existing batch results
    processed_files, file_results, ocr_cache = load_batch_results(batch_id)

    processed_pages = processed_files.get(filename, set())
    results = file_results.get(filename, [])

    if file_cache_key in ocr_cache:
        print(f"Using cached OCR results for {filename}")
        pages = ocr_cache[file_cache_key]
    else:
        print(f"Running OCR for {filename}")
        # Run OCR
        if ocr_method == "mistral":
            ocr_method_obj = ocr_load.MistralOCR()
            ocr = ocr_load.OCR(filepath, ocr_method_obj)
            ocr_results = ocr.get_text_blocks()
            pages = ocr_results
            print("========mistral OCR results==========")
            print(json.dumps(pages[0], indent=4))
        elif ocr_method == "textractocr":
            ocr_method_obj = ocr_load.TextractOCR()
            ocr = ocr_load.OCR(filepath, ocr_method_obj)
            ocr_results = ocr.get_text_blocks()
            pages = ocr_results
            print("========textractocr OCR results==========")
            print(json.dumps(pages[0]['value'], indent=4))
        elif ocr_method == "textract-kv":
            kv_extractor = KVExtractDocument(filepath)
            pages = kv_extractor.pages
            print("========textract-kv Image-to-text results==========")
            print(pages[0].kvs)
        else:
            raise ValueError(f"Unsupported OCR method: {ocr_method}")
        
        # Cache OCR results
        ocr_cache[file_cache_key] = pages
    
    # kv_document.pages or ocr_results
    if len(processed_pages) == len(pages):
        print(f"All pages already processed for {filename}")
        return results
    
    if extractor is None:
        extractor = OptimizedExtraction(fields=fields, model_type=model_type, max_attempts=max_attempts)
    
    # Process each page
    # kv_document.pages or ocr_results
    for page_num, page in enumerate(pages):
        if page_num in processed_pages:
            print(f"Skipping already processed page {page_num + 1} in {filename}")
            continue
            
        try:
            page_result = extractor.fv_extract(
                kvs=page.kvs if ocr_method == "textract-kv" else None,
                ocr=page["value"] if ocr_method != "textract-kv" else None,
                coordinates=page["words"] if spatial else None
            )
            print(f"Extracted fields for page {page_num + 1}: {page_result}")

            processed_pages.add(page_num)
            results.append(page_result)
            
            processed_files[filename] = processed_pages
            file_results[filename] = results
            if save_batch:
                save_batch_results(batch_id, processed_files, file_results, ocr_cache)
                
            print(f"Processed page {page_num + 1} of {filename}")
            
        except Exception as e:
            print(f"Error processing page {page_num + 1} in {filepath}: {str(e)}")
            raise e
        
    return results
    
def process_single_file_parallel(args):
    """Enhanced worker function for parallel processing with better error handling."""
    filepath, filename, fields, batch_id, model_type, ocr_method, spatial, extractor = args
    
    print(f"Starting parallel processing of {filename}")
    start_time = time.time()
    
    try:
        result = extract_file(
            filepath=filepath,
            fields=fields,
            batch_id=batch_id,
            filename=filename,
            model_type=model_type,
            ocr_method=ocr_method,
            spatial=spatial,
            extractor=extractor
        )
        
        processing_time = time.time() - start_time
        print(f"Completed {filename} in {processing_time:.2f} seconds")
        return filename, result, None
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"Error processing {filename} after {processing_time:.2f} seconds: {str(e)}")
        return filename, None, e

def extract_dir(dirpath: str, fields=w2_fields, file_out=None, batch_id: str=None, model_type: str="gpt-4.1", ocr_method: str="textract-kv", spatial: bool=False, prompt_opt: bool=False, label_file: str=None, test_file: str=None, test_label: str=None, max_workers: int=4) -> pd.DataFrame:
    '''
    Extracts W2 fields from all PDF/JPG/PNG files in a directory.
    
    Parameters:
    dirpath (str): The path to the directory containing PDF/JPG/PNG files.
    file_out (str): The path to save the output CSV or Excel file.
    fields (dict): A dictionary containing field descriptions, types, and examples.
    
    Return:
    pd.DataFrame: A DataFrame containing the extracted fields for each page.
    '''
    if batch_id is None:
        batch_id = get_batch_id_dir(dirpath)
    
    # List all PDF/JPG/PNG files in the directory
    files = [f for f in os.listdir(dirpath) if f.endswith('.pdf') or f.endswith('.png') or f.endswith('.jpg')]
    paths = [os.path.join(dirpath, f) for f in files]
    
    processed_files, file_results, ocr_cache = load_batch_results(batch_id)
    
    pending_files = []
    pending_paths = []
    completed_results = {}
        
    for filepath, filename in zip(paths, files):
        if filename in file_results and filename in processed_files:
            # Check if file is completely processed by seeing if we have results
            if file_results[filename]:  # Has some results
                print(f"File {filename} already processed, skipping")
                completed_results[filename] = file_results[filename]
                continue
        
        pending_files.append(filename)
        pending_paths.append(filepath)

    print(f"Processing {len(pending_files)} files ({len(files) - len(pending_files)} already completed)")

    base_extractor = OptimizedExtraction(fields=fields, model_type=model_type) if prompt_opt else None
    
    # Prepare arguments for parallel processing
    process_args = [
        (filepath, filename, fields, batch_id, model_type, ocr_method, spatial, base_extractor)
        for filepath, filename in zip(pending_paths, pending_files)
    ]

    new_results = {}
    failed_files = []

    if process_args:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file_parallel, args): args[1] 
                for args in process_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    filename, result, error = future.result()
                    if error is None and result is not None:
                        new_results[filename] = result
                        print(f"✓ Successfully processed {filename}")
                    else:
                        failed_files.append((filename, error))
                        print(f"✗ Failed to process {filename}: {error}")
                except Exception as e:
                    failed_files.append((filename, e))
                    print(f"✗ Exception processing {filename}: {e}")
    
    # Combine completed and new results
    all_results = {**completed_results, **new_results}
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print(f"\nSuccessfully processed {len(all_results)} files total")
    
    # Combine results into a single list of dictionaries
    entries = []
    for filename in files:
        if filename in all_results:
            file_results = all_results[filename]
            for pnum, page in enumerate(file_results):
                temp = {'file': filename, 'page': pnum + 1}
                temp.update(page)
                entries.append(temp)

    # Optionally convert to DataFrame and save as CSV
    # df = pd.DataFrame.from_dict(entries, orient='index')
    df = pd.DataFrame(entries)
    print(df)

    if prompt_opt and label_file:
        print("Running prompt optimization with evaluation...")

        test_files = [f for f in os.listdir(test_file) if f.endswith('.pdf') or f.endswith('.png') or f.endswith('.jpg')]
        test_paths = [os.path.join(test_file, f) for f in test_files]
        
        # Create temporary directory for intermediate results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save initial results
            temp_preds_file = os.path.join(temp_dir, "temp_preds.csv")
            df.to_csv(temp_preds_file, index=False)
            
            # First evaluation
            total_comparisons, correct_comparisons, mismatches = compare_datasets(
                labels_files=[label_file],
                preds_file=temp_preds_file
            )
            print(f"Initial accuracy: {correct_comparisons/total_comparisons:.2%}")
            print(f"Initial mismatches: {len(mismatches)}")
            grouped_mismatches = group_mismatches_by_file(mismatches)

            file_mismatches = {}
            for filename, mismatches in grouped_mismatches.items():
                cached_data = get_cached_file_data(filename, batch_id, ocr_method)
                file_mismatches[filename] = (mismatches, cached_data)
                
            base_extractor.add_iteration_mismatches(file_mismatches)
            
            # Second run with mismatches
            print("Running second extraction with mismatches...")
            clear_batch_results(batch_id)
            
            process_args = [
                (filepath, filename, fields, batch_id, model_type, ocr_method, spatial, base_extractor)
                for filepath, filename in zip(test_paths, test_files)
            ]
            
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_file_parallel, args): args[1] 
                    for args in process_args
                }
                
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        filename, result, error = future.result()
                        if error is None and result is not None:
                            results[filename] = result
                    except Exception as e:
                        print(f"Error in second run for {filename}: {e}")
            
            # Combine second run results
            entries = []
            for filename in files:
                if filename in results:
                    for pnum, page in enumerate(results[filename]):
                        temp = {'file': filename, 'page': pnum + 1}
                        temp.update(page)
                        entries.append(temp)
            
            df = pd.DataFrame(entries)
            df.to_csv(temp_preds_file, index=False)
            
            # Second evaluation
            total_comparisons, correct_comparisons, new_mismatches = compare_datasets(
                labels_files=[label_file],
                preds_file=temp_preds_file
            )
            print(f"Second run accuracy: {correct_comparisons/total_comparisons:.2%}")
            print(f"Second run mismatches: {len(new_mismatches)}")
            print(new_mismatches)

            new_grouped_mismatches = group_mismatches_by_file(new_mismatches)

            new_file_mismatches = {}
            for filename, mismatches in new_grouped_mismatches.items():
                cached_data = get_cached_file_data(filename, batch_id, ocr_method)
                new_file_mismatches[filename] = (mismatches, cached_data)

            base_extractor.add_iteration_mismatches(new_file_mismatches)
            
            # Final run with updated mismatches
            print("Running final extraction with updated mismatches...")
            clear_batch_results(batch_id)
            
            process_args = [
                (filepath, filename, fields, batch_id, model_type, ocr_method, spatial, base_extractor)
                for filepath, filename in zip(test_paths, test_files)
            ]
            
            results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(process_single_file_parallel, args): args[1] 
                    for args in process_args
                }
                
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        filename, result, error = future.result()
                        if error is None and result is not None:
                            results[filename] = result
                    except Exception as e:
                        print(f"Error in final run for {filename}: {e}")
            
            # Combine final run results
            entries = []
            for filename in test_files:
                if filename in results:
                    for pnum, page in enumerate(results[filename]):
                        temp = {'file': filename, 'page': pnum + 1}
                        temp.update(page)
                        entries.append(temp)
            
            df = pd.DataFrame(entries)
            df.to_csv(temp_preds_file, index=False)
            
            # Final evaluation
            total_comparisons, correct_comparisons, mismatches = compare_datasets(
                labels_files=[test_label],
                preds_file=temp_preds_file
            )
            print(f"Final accuracy: {correct_comparisons/total_comparisons:.2%}")
    
    return df

def extract_zip(file, fields=w2_fields, file_out=None) -> pd.DataFrame:
    '''
    Extracts W2 fields from a ZIP file containing PDF files.
    
    Parameters:
    filepath (str): The ZIP file object.
    file_out (str): The path to save the output CSV or Excel file.
    fields (dict): A dictionary containing field descriptions, types, and examples.
    
    Return:
    pd.DataFrame: A DataFrame containing the extracted fields for each page.
    '''
    filepath = file.name
    batch_id = get_batch_id_zip(file)
    # Create a temporary directory using tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Extract the uploaded ZIP file into the temporary directory
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(os.path.join(temp_dir, filepath))

        uploaded_path = os.path.join(temp_dir, filepath)
        
        w2_dir = [os.path.join(uploaded_path, f) for f in os.listdir(uploaded_path) if not f.startswith('__')][0]
        print("Extracting from:", w2_dir)

        # Run your processing function on the extracted files
        result_df = extract_dir(w2_dir, fields, file_out, batch_id)

    # Convert DataFrame to excel for download
    bytes_io = BytesIO()
    with pd.ExcelWriter(bytes_io, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']

            # Adjust column widths
            for col_idx, col in enumerate(result_df.columns, start=1):
                max_length = max(result_df[col].astype(str).map(len).max(), len(col)) + 2  # Add padding
                worksheet.set_column(col_idx - 1, col_idx - 1, max_length)
    bytes_io.seek(0)  # Move the cursor to the beginning of the BytesIO object
    
    return bytes_io

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract fields from PDF/JPG/PNG files.")
    parser.add_argument("--fieldpath", type=str, required=True, help="Path to the field definitions file (.json, .yaml, or .yml)")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the directory, single PDF file, or ZIP file containing PDFs")
    
    parser.add_argument("--type", default=None, choices=['file', 'dir', 'zip'], type=str, help="Type of extraction: 'file' for single file, 'dir' for directory, 'zip' for zip file.")
    parser.add_argument("--file_out", default=None, type=str, help="Path to save the output CSV or Excel file. None to skip saving.")
    parser.add_argument("--max_attempts", default=3, type=int, help="Maximum number of attempts to extract fields.")
    parser.add_argument("--model_type_or_path", default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-o3"], help="Model type used for fv extraction")
    parser.add_argument("--ocr_method", default="textract-kv", choices=["mistral", "textractocr", "textract-kv"], help="OCR method to use")
    parser.add_argument("--spatial_ocr", action="store_true", help="Enable spatial OCR with coordinate information")
    parser.add_argument("--prompt_opt", action="store_true", help="Enable prompt optimization with evaluation")
    parser.add_argument("--label_file", type=str, help="Path to the label file for evaluation when prompt_opt is True")
    parser.add_argument("--test_file", type=str, help="Path to the test file path for evaluation when prompt_opt is True")
    parser.add_argument("--test_label", type=str, help="Path to the test label file for evaluation when prompt_opt is True")
    parser.add_argument("--save_batch", action="store_true", help="Save batch results")
    args = parser.parse_args()

    try:
        # Create a file-like object for the field definitions file
        with open(args.fieldpath, 'rb') as f:
            field_file = BytesIO(f.read())
            field_file.name = os.path.basename(args.fieldpath)
            custom_fields = load_fields(field_file)
    except Exception as e:
        print(f"Error loading field definitions: {str(e)}")
        exit(1)

    # predict type based on file extension if not specified
    if args.type is None:
        if args.filepath.endswith('.zip'):
            args.type = 'zip'
        elif args.filepath.endswith('.pdf') or args.filepath.endswith('.png') or args.filepath.endswith('.jpg'):
            args.type = 'file'
        else:
            args.type = 'dir'
    
    # Call the extract_all function with the provided arguments
    try:
        if args.type == 'file':
            # If a single PDF file is provided, extract from that file
            filename = os.path.basename(args.filepath)
            result_df = extract_file(args.filepath, custom_fields, args.max_attempts, batch_id=0, filename=filename, model_type=args.model_type_or_path, ocr_method=args.ocr_method, spatial=args.spatial_ocr, save_batch=args.save_batch)
        elif args.type == 'dir':
            # If a directory is provided, extract from all PDF files in that directory
            result_df = extract_dir(args.filepath, custom_fields, args.file_out, model_type=args.model_type_or_path, ocr_method=args.ocr_method, spatial=args.spatial_ocr, prompt_opt=args.prompt_opt, label_file=args.label_file, test_file=args.test_file, test_label=args.test_label)
        elif args.type == 'zip':
            # If a ZIP file is provided, extract from all PDF files in that ZIP file
            with open(args.filepath, 'rb') as f:
                zip_file = BytesIO(f.read())
                zip_file.name = os.path.basename(args.filepath)
                result_df = extract_zip(zip_file, custom_fields, args.file_out)
        print("Processing completed successfully!")
        print(result_df)
        # Save to CSV or Excel if specified
        if args.file_out is not None:
            # CSV
            if args.file_out.endswith('.csv'):
                result_df.to_csv(args.file_out) 
            
            # Excel
            elif args.file_out.endswith('.xlsx'):
                # Use xlsxwriter to save DataFrame to Excel
                with pd.ExcelWriter(args.file_out, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Sheet1')
                    worksheet = writer.sheets['Sheet1']

                    # Adjust column widths
                    for col_idx, col in enumerate(result_df.columns, start=1):
                        max_length = max(result_df[col].astype(str).map(len).max(), len(col)) + 2  # Add padding
                        worksheet.set_column(col_idx - 1, col_idx - 1, max_length)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error extracting fields: {str(e)}")
        exit(1)
