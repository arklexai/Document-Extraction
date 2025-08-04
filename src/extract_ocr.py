from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import tempfile
import argparse
from typing import List
import pandas as pd
import yaml

from example import Example
from fv_extract import FVExtraction
from evaluation import compare_datasets
from optimizer import PromptOptimizer


def load_fields_from_path(filepath: str) -> dict[str, dict]:
    """Load field definitions from a file path (JSON or YAML)."""
    _, ext = os.path.splitext(filepath)
    
    with open(filepath, 'r') as f:
        if ext.lower() in ['.yaml', '.yml']:
            custom_fields = yaml.safe_load(f)
        elif ext.lower() == '.json':
            custom_fields = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .json, .yaml, or .yml")
    
    return custom_fields


def load_fields(file) -> dict[str, dict]:
    """Load field definitions from a file-like object (for compatibility)."""
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


def create_trainset(ocr_cache, labels_df, fields, ocr_mode: str="ocr") -> List[Example]:
    """Create training dataset from OCR cache and labels."""
    dataset = []
    
    for idx, row in labels_df.iterrows():
        filename = row['file']
        
        # Check if we have OCR data for this file
        if filename not in ocr_cache:
            print(f"Warning: No OCR data found for {filename}")
            continue

        ocr_pages = ocr_cache[filename]
        
        # Create expected output dictionary
        expected_fields = {}
        for field in fields.keys():
            value = row[field]
            expected_fields[field] = value
        
        example = Example(
            fields=str(fields),
            context=str(ocr_pages),
            mode=ocr_mode,
            output=expected_fields
        )
        
        dataset.append(example)
        print(f"Added training example for {filename}")
    
    return dataset


def process_single_ocr_data(filename_and_data):
    """Process a single OCR data entry with the extractor."""
    filename, ocr_data, extractor = filename_and_data
    try:
        # Handle different OCR data structures
        if isinstance(ocr_data, list):
            if len(ocr_data) == 1:
                ocr_content = ocr_data[0].get('content', str(ocr_data[0]))
            else:
                ocr_content = '\n'.join([
                    item.get('content', str(item)) if isinstance(item, dict) else str(item) 
                    for item in ocr_data
                ])
        elif isinstance(ocr_data, dict):
            ocr_content = ocr_data.get('content', str(ocr_data))
        else:
            ocr_content = str(ocr_data)
        
        # Extract using the extractor
        result = extractor.fv_extract(ocr=ocr_content, include_rationale=False)
        
        return filename, [result], None
        
    except Exception as e:
        return filename, None, str(e)


def optimize_prompt_from_external_data(
        fields: dict[str, dict],
        test_file: str,
        test_label: str = None,
        training_file: str = None,
        training_label: str = None,
        file_out: str = None,
        model_type: str="gpt-4o-mini",
        prompt_opt: bool=False,
        opt_iterations: int = 3,
        max_workers: int=4
    ):
    """Optimize prompts using external OCR data and evaluate on test data."""
    base_extractor = FVExtraction(fields=fields, model_type=model_type)
    extractor_to_use = base_extractor

    if prompt_opt and training_file and training_label:
        print("Loading training OCR data...")
        try:
            with open(training_file, 'r') as f:
                external_ocr_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load external OCR data from {training_file}: {e}")
        
        # Load training labels
        print("Loading training labels...")
        try:
            labels_df = pd.read_csv(training_label, keep_default_na=False, na_filter=False)
        except Exception as e:
            raise ValueError(f"Failed to load training labels from {training_label}: {e}")
        
        print("Creating training dataset...")
        dataset = create_trainset(
            ocr_cache=external_ocr_data,
            labels_df=labels_df, 
            fields=fields
        )

        print(f"Optimizing prompts with {len(dataset)} training examples...")
        optimizer = PromptOptimizer(fields=fields, model_type=model_type)
        optimized_extractor = optimizer.optimize_prompt(
            extractor=base_extractor, 
            train_data=dataset, 
            iterations=opt_iterations
        )
        extractor_to_use = optimized_extractor
        print("Prompt optimization completed!")

    print("Loading test OCR data...")
    try:
        with open(test_file, 'r') as f:
            test_ocr_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load test OCR data from {test_file}: {e}")
    
    print(f"Processing {len(test_ocr_data)} test files...")
    results = {}
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_filename = {
            executor.submit(process_single_ocr_data, (filename, ocr_data, extractor_to_use)): filename
            for filename, ocr_data in test_ocr_data.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                filename, result, error = future.result()
                if error is None and result is not None:
                    results[filename] = result
                    print(f"✓ Successfully processed {filename}")
                else:
                    failed_files.append((filename, error))
                    print(f"✗ Failed to process {filename}: {error}")
            except Exception as e:
                failed_files.append((filename, e))
                print(f"✗ Exception processing {filename}: {e}")
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    print(f"Successfully processed {len(results)} files")

    # Convert results to DataFrame
    entries = []
    for filename, file_results in results.items():
        if file_results:
            for pnum, page_result in enumerate(file_results):
                temp = {'file': filename, 'page': pnum + 1}
                temp.update(page_result.get('result', {}))
                entries.append(temp)
    
    df = pd.DataFrame(entries)
    print("Test results:")
    print(df)

    # Save results if output file specified
    if file_out is not None:
        if file_out.endswith('.csv'):
            df.to_csv(file_out, index=False)
        elif file_out.endswith('.xlsx'):
            with pd.ExcelWriter(file_out, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                worksheet = writer.sheets['Sheet1']
                
                # Adjust column widths
                for col_idx, col in enumerate(df.columns, start=1):
                    max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(col_idx - 1, col_idx - 1, max_length)
        print(f"Test results saved to: {file_out}")

    # Evaluate against test labels if provided
    if test_label:
        print("Evaluating test results...")
        try:
            total_comparisons, correct_comparisons, mismatches = compare_datasets(
                labels_files=[test_label],
                preds_file=file_out if file_out else None,
                preds_df=df if file_out is None else None
            )
            accuracy = correct_comparisons / total_comparisons if total_comparisons > 0 else 0
            print(f"Test accuracy: {accuracy:.2%}")
            print(f"Total comparisons: {total_comparisons}")
            print(f"Correct: {correct_comparisons}")
            print(f"Mismatches: {len(mismatches)}")
        except Exception as e:
            print(f"Warning: Could not evaluate results: {e}")

    return df


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Extract fields from OCR data using prompt optimization.")
    
    # Required arguments
    parser.add_argument("--fieldpath", type=str, required=True, 
                       help="Path to the field definitions file (.json, .yaml, or .yml)")
    parser.add_argument("--test_file", type=str, required=True, 
                       help="Path to the test OCR data file (.json)")
    
    # Optional arguments
    parser.add_argument("--test_label", type=str, default=None,
                       help="Path to the test labels file (.csv) for evaluation")
    parser.add_argument("--training_file", type=str, default=None,
                       help="Path to the training OCR data file (.json) for prompt optimization")
    parser.add_argument("--training_label", type=str, default=None,
                       help="Path to the training labels file (.csv) for prompt optimization")
    parser.add_argument("--file_out", type=str, default=None,
                       help="Path to save the output CSV or Excel file")
    parser.add_argument("--model_type", default="gpt-4o-mini", 
                       choices=["gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-o3", 
                               "Qwen/Qwen2.5-7B-Instruct-Turbo", "Qwen/Qwen3-8B", 
                               "mistralai/Mistral-7B-Instruct-v0.2"],
                       help="Model type used for field extraction")
    parser.add_argument("--prompt_opt", action="store_true",
                       help="Enable prompt optimization (requires training_file and training_label)")
    parser.add_argument("--opt_iterations", default=3, type=int,
                       help="Number of optimization iterations")
    parser.add_argument("--max_workers", default=4, type=int,
                       help="Number of worker threads for parallel processing")

    args = parser.parse_args()

    # Validate arguments
    if args.prompt_opt and (not args.training_file or not args.training_label):
        parser.error("--prompt_opt requires both --training_file and --training_label")

    try:
        # Load field definitions
        print(f"Loading field definitions from {args.fieldpath}")
        custom_fields = load_fields_from_path(args.fieldpath)
        print(f"Loaded {len(custom_fields)} field definitions")
        
        # Run extraction with optimization
        result_df = optimize_prompt_from_external_data(
            fields=custom_fields,
            test_file=args.test_file,
            test_label=args.test_label,
            training_file=args.training_file,
            training_label=args.training_label,
            file_out=args.file_out,
            model_type=args.model_type,
            prompt_opt=args.prompt_opt,
            opt_iterations=args.opt_iterations,
            max_workers=args.max_workers
        )
        
        print("Processing completed successfully!")
        print(f"Final results shape: {result_df.shape}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during extraction: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()