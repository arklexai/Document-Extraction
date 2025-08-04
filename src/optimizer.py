import pandas as pd
from typing import List, Dict, Tuple

from pydantic import BaseModel, Field
import tqdm

from example import Example
from fv_extract import FVExtraction
from evaluation import values_match
from model import get_model

class MetaPromptModule(BaseModel):
    tag: str = Field(..., min_length=1)
    instruction: List[str] = Field(..., min_length=3, max_length=8)

class MetaPromptOutput(BaseModel):
    response: List[MetaPromptModule]


class PromptOptimizer:
    """
    Separate class for handling prompt optimization using training data and labels.
    Extracts the optimization logic from FVExtraction and w2_extract.py.
    """
    
    def __init__(self, fields: dict[str, dict], model_type: str = "gpt-4o-mini", ):
       self.fields = fields
       self.model_type = model_type

       self.llm = get_model(model_type)

    def split_data(
        self, 
        train_data: List[Example],
        train_ratio: float = 0.7,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and evaluation sets.
        
        Args:
            ocr_data: List of OCR data dictionaries
            label_file: Path to labels CSV file
            
        Returns:
            Tuple of (train_ocr_data, eval_ocr_data, train_labels_df, eval_labels_df)
        """
        import random

        available_data = [data for data in train_data]
        
        if len(available_data) == 0:
            raise ValueError("No files found with corresponding labels")
        
        # Split data
        random.seed(random_seed)
        random.shuffle(available_data)
        
        train_size = int(len(available_data) * train_ratio)
        train_data = available_data[:train_size]
        eval_data = available_data[train_size:]
        
        # Split labels
        train_df = [data for data in train_data]
        eval_df = [data for data in eval_data]
        
        print(f"Split: {len(train_df)} training files, {len(eval_df)} evaluation files")
        
        return train_df, eval_df
    
    def evaluate_with_demos(self, example_data: List[Example]):
        total_comparisons, correct_comparisons, positive_cases, mismatches = 0, 0, [], []
        field_analysis = {}
        
        for data in example_data:
            pred, true, rationale = data._demos
            perfect_match = True

            for col in self.fields.keys():
                total_comparisons += 1
                if values_match(pred[col], true[col], col):
                    correct_comparisons += 1
                else:
                    mismatches.append({
                        'column': col,
                        'pred_value': pred[col],
                        'true_value': true[col],
                        'rationale': rationale
                    })
                    field_analysis[col] = field_analysis.get(col, 0) + 1
                    perfect_match = False

            if perfect_match:
                positive_cases.append({
                    'pred_value': pred,
                    'true_value': true,
                    'rationale': rationale
                })

        return total_comparisons, correct_comparisons, positive_cases, mismatches, field_analysis
    
    def evaluate_score(self, example_data: List[Example]):
        total_comparisons, correct_comparisons = 0, 0
        
        for data in example_data:
            pred, true, _ = data._demos

            for col in self.fields.keys():
                total_comparisons += 1
                if values_match(pred[col], true[col], col):
                    correct_comparisons += 1

        return total_comparisons, correct_comparisons
            
    
    def predict_from_example(self, extractor: FVExtraction, train_data: List[Example], max_workers: int = 4):
        """
        Extract fields from OCR data using the current extractor.
        
        Args:
            ocr_data: List of OCR data dictionaries
            spatial: Whether to use spatial coordinates
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
    
        def process_single_example(data: Example):
            """Process a single example in a worker thread."""
            try:
                ocr_context = data.context
                mode = data.mode
                label = data.labels().get('output', {})

                if mode == "kv":
                    page_result = extractor.fv_extract(kvs=ocr_context, include_rationale=True)
                elif mode == "ocr":
                    page_result = extractor.fv_extract(ocr=ocr_context, include_rationale=True)
                elif mode == "spatial":
                    page_result = extractor.fv_extract(coordinates=ocr_context, include_rationale=True)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                
                return data, (page_result['result'], label, page_result['rationale']), None
                
            except Exception as e:
                return data, None, str(e)
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(process_single_example, data): data 
                for data in train_data
            }

            # Process results as they complete
            completed = 0
            total = len(train_data)
            
            for future in tqdm.tqdm(as_completed(future_to_example), total=total, desc="Processing examples"):
                data = future_to_example[future]
                try:
                    data, result, error = future.result()
                    if error is None and result is not None:
                        data._demos = result
                        completed += 1
                    else:
                        print(f"Error processing example: {error}")
                except Exception as e:
                    print(f"Exception processing example: {e}")
            
            print(f"Successfully processed {completed}/{total} examples")
        

    def generate_meta_prompt_description(self, base_prompt_template: Dict, positive_cases: List[Example], mismatches: List[Dict], field_analysis: Dict):
        prompt = """
        You will generate **3 to 5 distinct meta‑prompt modules**—each with its own `tag` and a set of **3–8 actionable guidance strings**—to improve OCR field extraction accuracy.

        Each module should:
        1. Target a **different strategy** (e.g., regex normalization, semantic validation, fallback heuristics, cross‑field checks).
        2. Be concise: each guidance item must be **one to two sentences**.
        3. Be **actionable** and **testable** (e.g., “If the extracted date does not match YYYY‑MM‑DD, re‑parse with US format and validate chronological order”).

        Before generating, analyze:
        A. **Field Descriptions** for patterns, data types, typical formats.  
        B. **Positive Exemplars** to identify success drivers.  
        C. **Failed Exemplars & Error Analysis** to pinpoint recurring extraction issues.  
        D. **Prompt Limitations** and any systematic biases

        **Output Format** (JSON array):
        ```json
        [
            {{
                "tag": "<ShortTag>",
                "instruction": [
                    "...",
                    "...",
                    "..."
                ]
            }},
            // up to 5 modules
        ]
        ``` 

        ===Input===
        Field descriptions:
        {field}

        Base prompt:
        {base_prompt}

        Positive extraction exemplars:
        {positive_cases}

        Failure extraction exemplars (mismatches):
        {new_mismatches}

        Field error analysis:
        {field_analysis}

        ===End of Input===
        """
        formatted_prompt = prompt.format(
            field = self.fields,
            base_prompt = base_prompt_template,
            positive_cases = positive_cases,
            new_mismatches = mismatches,
            field_analysis = field_analysis
        )
        response = self.llm.get_response(prompt=formatted_prompt, output_format=MetaPromptOutput)

        return response

    
    def optimize_prompt(
        self, 
        extractor: FVExtraction,
        train_data: List[Example],
        iterations: int = 3
    ) -> FVExtraction:
        """
        Main optimization function that trains the prompt using mismatches.
        
        Args:
            train_data: Either directory path or pre-processed OCR data
            label_file: Path to labels CSV file
            ocr_method: OCR method to use
            spatial: Whether to use spatial coordinates
            
        Returns:
            Optimized FVExtraction instance
        """
        
        # Split into train/eval
        train_raw_df, eval_raw_df = self.split_data(train_data)

        best_accuracy = -1
        best_meta = None

        for i in range(iterations):
            print(f"\n=== Optimization Iteration {i + 1}/{iterations} ===")
        
            # Extract fields from OCR data
            self.predict_from_example(extractor, train_raw_df)

            # Evaluate
            total_comparisons, correct_comparisons, positive_cases, mismatches, field_analysis = self.evaluate_with_demos(train_raw_df)

            if best_accuracy == -1:
                best_accuracy = correct_comparisons / total_comparisons

            print(f"Accuracy: {correct_comparisons / total_comparisons * 100:.2f}%")
            print(f"Positive cases: {len(positive_cases)}")

            response = self.generate_meta_prompt_description(extractor.base_template_parts, positive_cases, mismatches, field_analysis)
            meta_prompts = response.response
            print(f"Number of Meta prompt descriptions: {len(meta_prompts)}")
            
        
            iteration_best_accuracy = best_accuracy
            iteration_best_meta = best_meta
            for meta in meta_prompts:
                extractor.meta_prompt = meta

                self.predict_from_example(extractor, eval_raw_df)

                total_comparisons, correct_comparisons = self.evaluate_score(eval_raw_df)

                accuracy = correct_comparisons / total_comparisons if total_comparisons else 0
                print(f"Accuracy: {accuracy * 100:.2f}%")
                if accuracy > iteration_best_accuracy:
                    iteration_best_accuracy = accuracy
                    iteration_best_meta = meta

            if iteration_best_accuracy > best_accuracy:
                best_accuracy = iteration_best_accuracy
                best_meta = iteration_best_meta

        extractor.meta_prompt = best_meta
        print(f"Best meta prompt: {best_meta}")
        
        return extractor