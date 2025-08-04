# Document-Extraction

## Setup
Python Version: `3.12.9`
Packages: `requirements.txt`

#### API Key
-  OCR
  
    **AWS API key**: Insert it here `/Users/[user]/.aws/credentials`, `.env`, or CLI \
    **Mistral API key**: Insert it here `.env` as `export MISTRAL_API_KEY=<api_key>`

- LLM

    Depends on the model you are using. Set the corresponding API key in `.env`

## Run
### Streamlit
Launching Streamlit: `streamlit run src/w2_app.py`

Import zipped directory of the PDFs

## Document Extraction with OCR Process
#### For working with OCR pipeline and optimizing prompts:
```bash
python src/w2_extract.py --fieldpath <fieldpath> --filepath <filepath> [options]
```

#### Required Arguments:
- `--fieldpath`: Path to the field definitions file (.json, .yaml, or .yml)
- `--filepath`: Path to directory, single PDF file, or ZIP file containing PDFs

#### Optional Arguments:
- `--type {file,dir,zip}`: Type of extraction (auto-detected if not specified)
- `--file_out FILE`: Path to save output CSV or Excel file
- `--max_attempts N`: Maximum number of extraction attempts (default: 3)
- `--model_type_or_path {gpt-4o-mini,gpt-4.1,gpt-o3}`: Model type for extraction (default: gpt-4o-mini)
- `--ocr_method {mistral,textractocr,textract-kv}`: OCR method to use (default: mistral)
- `--spatial_ocr`: Enable spatial OCR with coordinate information
- `--prompt_opt`: Enable prompt optimization with evaluation
- `--label_file FILE`: Path to label file for evaluation (required when using --prompt_opt)
- `--train_file FILE`: Path to the training file path for prompt optimization (required when using --prompt_opt)
- `--train_label FILE`: Path to the training label file for prompt optimization (required when using --prompt_opt)

Should be able to handle PDF, directories, and zipped directory paths

## OCR-based Extraction and Prompt Optimization
#### For working with pre-extracted OCR data and optimizing prompts:
```bash
python src/extract_ocr.py --fieldpath <fieldpath> --test_file <test_file> [options]
```

##### Required Arguments:
- `--fieldpath`: Path to the field definitions file (.json, .yaml, or .yml)
- `--test_file`: Path to the test OCR data file (.json)

##### Optional Arguments:
- `--test_label`: Path to the test labels file (.csv) for evaluation
- `--training_file`: Path to the training OCR data file (.json) for prompt optimization
- `--training_label`: Path to the training label file (.csv) for prompt optimization
- `--file_out`: Path to save the output CSV or Excel file
- `--model_type {gpt-4o-mini,gpt-4.1,gpt-o3}`: Model type for extraction (default: gpt-4o-mini)
- `--prompt_opt`: Enable prompt optimization (requires training_file and training_label)
- `--opt_iterations`: Number of optimization iterations (default: 3)
- `--max_workers`: Number of worker threads for parallel processing (default: 4)

## Evaluation

### Standalone Evaluation
Compare extraction results against ground truth labels using the evaluation script:

```bash
python src/evaluation.py --labels-filepath <ground_truth_files> --preds-filepath <predictions_file>
```

#### Required Arguments:
- `--labels-filepath`: One or more paths to ground truth label CSV files (can specify multiple files)
- `--preds-filepath`: Path to predictions CSV file (output from w2_extract.py)

#### Optional Arguments:
- `--evaluation-type`: Default: all ('all', 'exact_match', 'partial_match', 'f1', 'recall', 'precision')
- `--qualitative`: Enable qualitative mismatch analysis
- `--save-dir`: Directory to save evaluation results. If not provided, results will only be printed.
