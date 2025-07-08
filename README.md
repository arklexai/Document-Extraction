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

### Python
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

Should be able to handle PDF, directories, and zipped directory paths

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
