import pandas as pd
import re
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score
from typing import List, Dict, Any, Tuple
import json
import csv
from datetime import datetime


# Columns to compare
COLUMNS_TO_COMPARE = ['a', 'e', 'f', '1', '5', '12a', '12b', '14']

# Reference column for matching entries
REFERENCE_COLUMN = 'file'
# Threshold for partial match
PARTIAL_MATCH_THRESHOLD = 0.8

def normalize_string(text):
    if pd.isna(text):
        return text
    # Convert to string, remove extra spaces, and convert to lowercase
    text = re.sub(r'\s+', ' ', str(text).strip())
    return text.lower()

def normalize_ssn(ssn):
    if pd.isna(ssn):
        return ssn
    
    ssn_str = re.sub(r'[\s-]', '', str(ssn).strip())

    return ssn_str.lower()

def normalize_address(addr):
    if pd.isna(addr):
        return addr
    # Normalize spaces and case
    # addr = normalize_string(addr)
    addr = str(addr).lower().strip()

     # Add spaces around punctuation before removing it to prevent words from merging
    addr = re.sub(r'([^\w\s])(?=\w)', r'\1 ', addr)
    addr = re.sub(r'(\w)(?=[^\w\s])', r'\1 ', addr)

    # Remove spaces around punctuation
    addr = re.sub(r'[^\w\s]', '', addr)

    addr = re.sub(r'\s+', ' ', addr).strip()

    return addr

def exact_match(input_value_1, input_value_2) -> bool:
    """
    Check if two values are exactly the same after normalization.
    
    Args:
        input_value_1: First value to compare
        input_value_2: Second value to compare
    
    Returns:
        bool: True if values match exactly, False otherwise
    """
    # Handle NaN values
    if pd.isna(input_value_1) and pd.isna(input_value_2):
        return True
    if pd.isna(input_value_1) or pd.isna(input_value_2):
        return False
    
    norm_val1 = normalize_string(input_value_1)
    norm_val2 = normalize_string(input_value_2)
    
    return norm_val1 == norm_val2

def longest_common_sequence(input_value_1, input_value_2) -> float:
    """
    Calculate the longest common subsequence ratio between two strings.
    Used to calculate partial match percentage.
    
    Args:
        input_value_1: First string to compare
        input_value_2: Second string to compare
    
    Returns:
        float: Ratio of LCS length to the length of the longer string (0.0 to 1.0)
    """
    # Handle NaN values
    if pd.isna(input_value_1) and pd.isna(input_value_2):
        return 1.0
    if pd.isna(input_value_1) or pd.isna(input_value_2):
        return 0.0
    
    str1 = normalize_string(input_value_1)
    str2 = normalize_string(input_value_2)
    
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    max_length = max(len(str1), len(str2))
    
    return lcs_length / max_length if max_length > 0 else 0.0

def f1(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    """Wrapper for sklearn f1_score."""
    return f1_score(y_true, y_pred, **kwargs)

def recall(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    """Wrapper for sklearn recall_score."""
    return recall_score(y_true, y_pred, **kwargs)

def precision(y_true: List[int], y_pred: List[int], **kwargs) -> float:
    """Wrapper for sklearn precision_score."""
    return precision_score(y_true, y_pred, **kwargs)

def eval(pred: List[str], actual: List[str], eval_type: str = "all", qualitative: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation function for comparing predicted and actual values.
    
    Args:
        pred: List of predicted values
        actual: List of actual/ground truth values
        eval_type: Type of evaluation ("all", "exact_match", "partial_match", "f1", "recall", "precision")
        qualitative: Whether to return qualitative mismatch results
    
    Returns:
        Dict containing evaluation metrics and optionally qualitative results
    """
    if len(pred) != len(actual):
        raise ValueError(f"Prediction and actual lists must have same length. Got {len(pred)} and {len(actual)}")
    
    results = {}
    
    # Calculate exact matches
    if eval_type in ["all", "exact_match", "f1", "recall", "precision"]:
        exact_matches = [exact_match(p, a) for p, a in zip(pred, actual)]
        exact_match_count = sum(exact_matches)
        exact_match_ratio = exact_match_count / len(pred) if len(pred) > 0 else 0.0
        
        if eval_type in ["all", "exact_match"]:
            results["exact_match_accuracy"] = exact_match_ratio
            results["exact_match_count"] = exact_match_count
            results["total_comparisons"] = len(pred)
    
    # Calculate partial matches
    if eval_type in ["all", "partial_match", "f1", "recall", "precision"]:
        lcs_scores = [longest_common_sequence(p, a) for p, a in zip(pred, actual)]
        avg_lcs_score = np.mean(lcs_scores) if lcs_scores else 0.0
        
        # Define partial match threshold
        partial_match_threshold = PARTIAL_MATCH_THRESHOLD
        partial_matches = [score >= partial_match_threshold for score in lcs_scores]
        partial_match_count = sum(partial_matches)
        partial_match_ratio = partial_match_count / len(pred) if len(pred) > 0 else 0.0
        
        if eval_type in ["all", "partial_match"]:
            results["partial_match_accuracy"] = partial_match_ratio
            results["partial_match_count"] = partial_match_count
            results["average_lcs_score"] = avg_lcs_score
    
    # Calculate F1, Recall, Precision if needed
    if eval_type in ["all", "f1", "recall", "precision"]:
        # Convert boolean matches to binary (1 for match, 0 for no match)
        y_true = [1] * len(pred)
        
        if "exact_match_accuracy" in results or eval_type in ["f1", "recall", "precision"]:
            y_pred_exact = [1 if match else 0 for match in exact_matches]
            
            if eval_type in ["all", "f1"]:
                results["exact_f1"] = f1(y_true, y_pred_exact, zero_division=0)
            if eval_type in ["all", "recall"]:
                results["exact_recall"] = recall(y_true, y_pred_exact, zero_division=0)
            if eval_type in ["all", "precision"]:
                results["exact_precision"] = precision(y_true, y_pred_exact, zero_division=0)
        
        if "partial_match_accuracy" in results or eval_type in ["f1", "recall", "precision"]:
            y_pred_partial = [1 if match else 0 for match in partial_matches]
            
            if eval_type in ["all", "f1"]:
                results["partial_f1"] = f1(y_true, y_pred_partial, zero_division=0)
            if eval_type in ["all", "recall"]:
                results["partial_recall"] = recall(y_true, y_pred_partial, zero_division=0)
            if eval_type in ["all", "precision"]:
                results["partial_precision"] = precision(y_true, y_pred_partial, zero_division=0)
    
    # Add qualitative results if requested
    if qualitative and (eval_type in ["all", "exact_match"] or eval_type in ["f1", "recall", "precision"]):
        mismatches = []
        for i, (p, a) in enumerate(zip(pred, actual)):
            if not exact_matches[i]:
                mismatches.append({
                    'index': i,
                    'predicted': p,
                    'actual': a,
                    'lcs_score': lcs_scores[i] if 'lcs_scores' in locals() else longest_common_sequence(p, a),
                    'exact_match': exact_matches[i],
                    'partial_match': partial_matches[i] if 'partial_matches' in locals() else longest_common_sequence(p, a) >= 0.5
                })
        results["mismatches"] = mismatches
        results["mismatch_count"] = len(mismatches)
    
    return results

def values_match(val1, val2, column):
    # Handle NaN values
    if pd.isna(val1) and pd.isna(val2):
        return True
    if pd.isna(val1) or pd.isna(val2):
        return False
    
    # For SSN fields (column 'a')
    if column == 'a':
        return normalize_ssn(val1) == normalize_ssn(val2)
    
    # For address fields (column 'f')
    if column == 'f':
        return normalize_address(val1) == normalize_address(val2)
    
    # For numeric fields, try float comparison first
    if column in ['1', '5']:
        try:
            float_val1 = float(val1)
            float_val2 = float(val2)

            # Compare as floats with small tolerance for floating point precision
            return abs(float_val1 - float_val2) < 1e-9
        except (ValueError, TypeError):
            # If conversion fails, fall back to string comparison
            pass
    
    # For all other fields, normalize spaces and case
    return normalize_string(val1) == normalize_string(val2)
def compare_datasets(labels_files, preds_file):
    true_dfs = []
    for file in labels_files:
        df = pd.read_csv(file, converters={'a': str})
        true_dfs.append(df)
    true_df = pd.concat(true_dfs, ignore_index=True)


    # Read the predictions
    raw_df = pd.read_csv(preds_file, converters={'a': str})

    total_comparisons = 0
    correct_comparisons = 0
    mismatches = []

    # Compare entries based on reference column
    for idx, raw_row in raw_df.iterrows():
        true_row = true_df[true_df[REFERENCE_COLUMN] == raw_row[REFERENCE_COLUMN]]
        
        if len(true_row) == 0:
            print(f"No matching entry found in true data for name: {raw_row[REFERENCE_COLUMN]}")
            continue
        
        true_row = true_row.iloc[0]  # Get the first matching row
        
        for col in COLUMNS_TO_COMPARE:
            total_comparisons += 1
            if values_match(raw_row[col], true_row[col], col):
                correct_comparisons += 1
            else:
                mismatches.append({
                    'name': raw_row[REFERENCE_COLUMN],
                    'column': col,
                    'raw_value': raw_row[col],
                    'true_value': true_row[col]
                })

    return total_comparisons, correct_comparisons, mismatches

def align_datasets(labels_files, preds_file):
    """
    Align the prediction and ground truth datasets by matching rows using the reference column.
    
    Args:
        labels_files: List of paths to ground truth label CSV files
        preds_file: Path to predictions CSV file
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (aligned_pred_df, aligned_true_df)
            Both dataframes have the same rows (matched by reference column) and same column order
    """
    true_dfs = []
    for file in labels_files:
        df = pd.read_csv(file)
        true_dfs.append(df)
    true_df = pd.concat(true_dfs, ignore_index=True)

    pred_df = pd.read_csv(preds_file)
    
    aligned_pred_rows = []
    aligned_true_rows = []
    
    for idx, pred_row in pred_df.iterrows():
        matching_true_rows = true_df[true_df[REFERENCE_COLUMN] == pred_row[REFERENCE_COLUMN]]
        
        if len(matching_true_rows) > 0:
            true_row = matching_true_rows.iloc[0]  # Get the first matching row
            aligned_pred_rows.append(pred_row)
            aligned_true_rows.append(true_row)
    
    aligned_pred_df = pd.DataFrame(aligned_pred_rows).reset_index(drop=True)
    aligned_true_df = pd.DataFrame(aligned_true_rows).reset_index(drop=True)
    
    return aligned_pred_df, aligned_true_df

def main():
    parser = argparse.ArgumentParser(description='Compare raw labels with true dataset.')
    parser.add_argument('--labels-filepath', nargs='+', required=True,
                      help='Paths to ground truth label CSV files')
    parser.add_argument('--preds-filepath', required=True,
                      help='Path to predictions CSV file')
    parser.add_argument('--evaluation-type', default='all',
                      choices=['all', 'exact_match', 'partial_match', 'f1', 'recall', 'precision'],
                      help='Type of evaluation metrics to calculate')
    parser.add_argument('--qualitative', action='store_true',
                      help='Enable qualitative mismatch analysis')
    parser.add_argument('--save-dir', type=str,
                      help='Directory to save evaluation results. If not provided, results will only be printed.')
    
    args = parser.parse_args()

    for file_path in args.labels_filepath + [args.preds_filepath]:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Align datasets first
    aligned_pred_df, aligned_true_df = align_datasets(args.labels_filepath, args.preds_filepath)
    
    if len(aligned_pred_df) == 0:
        print("No matching entries found between prediction and ground truth datasets.")
        return
    
    print(f"Found {len(aligned_pred_df)} matching entries for evaluation.")
    
    qualitative = args.qualitative
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    for col in COLUMNS_TO_COMPARE:
        if col in aligned_pred_df.columns and col in aligned_true_df.columns:
            print(f"\n{'-'*20} Column: {col} {'-'*20}")
            
            pred_values = aligned_pred_df[col].fillna("").astype(str).tolist()
            true_values = aligned_true_df[col].fillna("").astype(str).tolist()
            
            col_results = eval(pred_values, true_values, args.evaluation_type, qualitative)
            
            if args.evaluation_type in ['all', 'exact_match']:
                if 'exact_match_accuracy' in col_results:
                    print(f"Exact Match Accuracy: {col_results['exact_match_accuracy']:.2%}")
            
            if args.evaluation_type in ['all', 'partial_match']:
                if 'partial_match_accuracy' in col_results:
                    print(f"Partial Match Accuracy: {col_results['partial_match_accuracy']:.2%}")
                    print(f"Average LCS Score: {col_results['average_lcs_score']:.3f}")
            
            if args.evaluation_type in ['all', 'f1']:
                if 'exact_f1' in col_results:
                    print(f"Exact Match - F1: {col_results['exact_f1']:.3f}")
                if 'partial_f1' in col_results:
                    print(f"Partial Match - F1: {col_results['partial_f1']:.3f}")
            
            if args.evaluation_type in ['all', 'recall']:
                if 'exact_recall' in col_results:
                    print(f"Exact Match - Recall: {col_results['exact_recall']:.3f}")
                if 'partial_recall' in col_results:
                    print(f"Partial Match - Recall: {col_results['partial_recall']:.3f}")
            
            if args.evaluation_type in ['all', 'precision']:
                if 'exact_precision' in col_results:
                    print(f"Exact Match - Precision: {col_results['exact_precision']:.3f}")
                if 'partial_precision' in col_results:
                    print(f"Partial Match - Precision: {col_results['partial_precision']:.3f}")
            
            if qualitative and 'mismatch_count' in col_results and col_results['mismatch_count'] > 0:
                print(f"Mismatches: {col_results['mismatch_count']}")
                print("Sample mismatches:")
                for i, mismatch in enumerate(col_results['mismatches'][:20]): 
                    print(f"  {i+1}. Predicted: '{mismatch['predicted']}' | "
                          f"Actual: '{mismatch['actual']}' | "
                          f"LCS Score: {mismatch['lcs_score']:.3f}")
                if len(col_results['mismatches']) > 3:
                    print(f"  ... and {len(col_results['mismatches'])-3} more")
    
    print(f"\n{'-'*20} Overall {'-'*20}")
    all_pred = []
    all_true = []
    
    for col in COLUMNS_TO_COMPARE:
        if col in aligned_pred_df.columns and col in aligned_true_df.columns:
            pred_values = aligned_pred_df[col].fillna("").astype(str).tolist()
            true_values = aligned_true_df[col].fillna("").astype(str).tolist()
            all_pred.extend(pred_values)
            all_true.extend(true_values)
    
    if all_pred and all_true:
        overall_results = eval(all_pred, all_true, args.evaluation_type, qualitative)
        
        if 'exact_match_accuracy' in overall_results:
            print(f"Overall Exact Match Accuracy: {overall_results['exact_match_accuracy']:.2%}")
        
        if 'partial_match_accuracy' in overall_results:
            print(f"Overall Partial Match Accuracy: {overall_results['partial_match_accuracy']:.2%}")
            print(f"Overall Average LCS Score: {overall_results['average_lcs_score']:.3f}")
        
        if args.evaluation_type in ['all', 'f1']:
            if 'exact_f1' in overall_results:
                print(f"Overall Exact Match - F1: {overall_results['exact_f1']:.3f}")
            if 'partial_f1' in overall_results:
                print(f"Overall Partial Match - F1: {overall_results['partial_f1']:.3f}")
        
        if args.evaluation_type in ['all', 'recall']:
            if 'exact_recall' in overall_results:
                print(f"Overall Exact Match - Recall: {overall_results['exact_recall']:.3f}")
            if 'partial_recall' in overall_results:
                print(f"Overall Partial Match - Recall: {overall_results['partial_recall']:.3f}")
        
        if args.evaluation_type in ['all', 'precision']:
            if 'exact_precision' in overall_results:
                print(f"Overall Exact Match - Precision: {overall_results['exact_precision']:.3f}")
            if 'partial_precision' in overall_results:
                print(f"Overall Partial Match - Precision: {overall_results['partial_precision']:.3f}")

    complete_results = {
        'metadata': {
            'labels_files': args.labels_filepath,
            'predictions_file': args.preds_filepath,
            'evaluation_type': args.evaluation_type,
            'total_entries': len(aligned_pred_df)
        },
        'columns': {},
        'overall': overall_results if all_pred and all_true else {}
    }

    # Add column-specific results
    for col in COLUMNS_TO_COMPARE:
        if col in aligned_pred_df.columns and col in aligned_true_df.columns:
            pred_values = aligned_pred_df[col].fillna("").astype(str).tolist()
            true_values = aligned_true_df[col].fillna("").astype(str).tolist()
            col_results = eval(pred_values, true_values, args.evaluation_type, qualitative)
            complete_results['columns'][col] = col_results

    # Save results if directory is specified
    if args.save_dir:
        # Create save directory
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_file = save_dir / f"evaluation_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(complete_results, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        
        if qualitative:
            mismatches_file = save_dir / f"mismatches_{timestamp}.csv"
            with open(mismatches_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Column', 'Index', 'Predicted', 'Actual', 'LCS_Score', 'Is_Partial_Match'])
                
                for col, results in complete_results['columns'].items():
                    if 'mismatches' in results:
                        for mismatch in results['mismatches']:
                            writer.writerow([
                                col,
                                mismatch['index'],
                                mismatch['predicted'],
                                mismatch['actual'],
                                mismatch['lcs_score'],
                                mismatch['partial_match']
                            ])
            print(f"Detailed mismatches saved to: {mismatches_file}")

if __name__ == '__main__':
    main() 
    
    # Evaluation with all metrics and qualitative analysis
    """
    python src/evaluation.py \
    --labels-filepath data/true_labels.csv \
    --preds-filepath data/raw_label.csv \
    --evaluation-type all \
    --qualitative \
    --save-dir results/w2
    """

    # Without qualitative analysis and use only evaluation type = exact_match
    """
    python src/evaluation.py \
    --labels-filepath data/true_labels.csv \
    --preds-filepath data/raw_label.csv \
    --evaluation-type exact_match
    """
    
    # Without qualitative analysis and use only evaluation type = recall
    """
    python src/evaluation.py \
    --labels-filepath data/true_labels.csv \
    --preds-filepath data/raw_label.csv \
    --evaluation-type recall
    """