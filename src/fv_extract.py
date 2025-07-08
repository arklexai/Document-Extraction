from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import tiktoken
load_dotenv()

import json
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, create_model
from typing import Optional, TypedDict, Union, Dict, List, Type

class FieldErrorAnalysis(TypedDict):
    expected: Union[str, float]
    actual: Union[str, float]
    reason: str
    filename: str

class ErrorAnalysis(TypedDict):
    """Model for structured error analysis output"""
    response: Dict[str, List[FieldErrorAnalysis]] 

class TypeMapping:
    STRING_TO_TYPE: Dict[str, Type] = {
        "string": str,
        "int": int,
        "float": float,
        "bool": bool,
    }

    @classmethod
    def string_to_type(cls, type_string: str) -> Type:
        """Convert a string representation to its corresponding Python type."""
        return cls.STRING_TO_TYPE.get(type_string)
    
class OptimizedExtraction:
    def __init__(self, fields: dict[str, dict], model_type: str="gpt-4o-mini", max_attempts: int = 3):
        self.fields = fields
        self.model_type = model_type
        self.max_attempts = max_attempts
        self.base_template_parts = self._initialize_base_template()
        
        self.iteration_mismatches = [] 
        self.iteration_error_analyses = []
        self.current_iteration = 0

        self.document_error_counts = defaultdict(int)
        self.field_error_counts = defaultdict(int) 

    def _initialize_base_template(self) -> Dict[str, str]:
        """Initialize the base prompt template parts."""
        return {
            'base': """
Extract value from a key:value list dictionary, OCR result, and/or coordinate layout given a description and type for each field in a `fields` dictionary.

Field names may not exactly match keys in the input data. Match values using semantic similarity, type expectations, and (when available) spatial relationships or OCR confidence.

Return key-value results in a JSON dictionary string, where:
- If a field is **not found**, return `null`.
- If a field is found but has **no value**, return an **empty string**.
- Values must respect their declared types (e.g., float, string).
- Output must be compatible with `json.loads()` (i.e., double-quoted strings).

{mode_instructions}
""",
            'kvs': """
Use only the key-value dictionary provided. Match field descriptions to dictionary keys using:
- Semantic similarity
- Field type expectations (e.g., float, string)
Normalize keys for comparison (e.g., lowercase, remove punctuation).
""",
            'ocr': """
Use the raw OCR text to identify values by:
- Finding lines that contain field-like descriptions
- Inferring label-value pairs by proximity in the OCR line order
- Matching expected data format or type (e.g., 9-digit number for SSN)
""",
            'spatial': """
Use OCR text and corresponding coordinates (x, y, confidence) to:
- Identify field-value pairs using spatial alignment (e.g., same row or column)
- Prefer high-confidence OCR results
- Choose the value closest in position and most similar in meaning to the field description
""",
            'error_header': "\n==Previous Extraction Errors - CRITICAL: Learn from these patterns==\n",
            'field_error': "\nField '{field}' ({desc}):\nPredicted (Extracted) Value:{pred}\nWhat went wrong:{reason}\nExpected (True) Value:{true}\n\n",
            'error_analysis_section': "",
            'example': """
==Example==
Example Fields dict:
{{
    'a': {{'desc': 'a: Employee's Social Security Number (SSN) [9 digits]', 'type': 'string', 'example': '123456789', 'required': True}},
    '1': {{'desc': '1 Wages tips other comp', 'type': 'float', 'example': 1234.56, 'required': True}},
    'e': {{'desc': 'e: Employee's Name', 'type': 'string', 'example': 'John Smith', 'required': True}},
    'f': {{'desc': 'f: Employee's Address', 'type': 'string', 'example': '123 Main St, City, State, ZIP', 'required': True}},
    '14': {{'desc': '14: Other', 'type': 'string', 'example': 'DCP-REG 916.18  CA-SDI 66.53', 'required': False}},
    '17': {{'desc': '17: Testing', "type": "string", 'example': '6,281.52', 'required': False}}
}}

Example Key-Value dict:
{{
    'a Employee's social security number'  : ['987654321 ', '987654321 ', '987654321 ', '987654321 ']
    'e/f Employee's name, address and ZIP code'  : ['Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ', 'Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ', 'Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ', 'Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ']
    '14 Other'  : ['', '', '', '']
}}

Example Return:
'''
{{
    'a': '987654321',
    'e': 'Yi Ding',
    'f': '6709 SETZLER PARKWAY BROOKLYN PARK MN 55445'
    '14': ''
    '17': null
}}
'''""",
            'input': """
==Input==
Input Fields dict: 
{fields}

Input Data (key:value list dictionary, OCR result, and/or coordinate layout):
{input_data}

Return:
'''
{{output}}
'''
"""
        }

    def generate_prompt(self, fields: dict, kvs: Dict = None, ocr: str = None, coordinates: List = None) -> tuple[str, int]:
        # Build prompt efficiently using string templates
        if kvs:
            mode_instructions = self.base_template_parts['kvs']
        elif ocr:
            mode_instructions = self.base_template_parts['ocr']
        elif ocr and coordinates:
            mode_instructions = self.base_template_parts['spatial']

        prompt_parts = [self.base_template_parts['base'].format(mode_instructions=mode_instructions)]

        if self.iteration_error_analyses:
            prompt_parts.append(self.base_template_parts['error_analysis_section'])

        # Add example section
        prompt_parts.append(self.base_template_parts['example'])

        input_data = []
        if kvs:
            input_data.append(f"Input Key-Value dict:\n{kvs}")
        if ocr:
            input_data.append(f"Input OCR result:\n{ocr}")
        if coordinates:
            input_data.append(f"Input coordinates:\n{coordinates}")
        
        # Add input section with formatted data
        prompt_parts.append(
            self.base_template_parts['input'].format(
                fields=fields,
                input_data='\n'.join(input_data)
            )
        )

        final_prompt = ''.join(prompt_parts)

        token_count = self.count_tokens(final_prompt)

        # Join all parts efficiently
        return final_prompt, token_count
    
    def count_tokens(self, prompt: str) -> int:
        """
        Count the number of tokens in a text string using the appropriate tokenizer for the model.
        
        Parameters:
        text (str): The text to count tokens for
        model_type (str): The model type to determine the appropriate tokenizer
        
        Returns:
        int: The number of tokens in the text
        """
        model_encodings = {
            "gpt-4o-mini": "o200k_base",
            "gpt-4o": "o200k_base", 
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-o3": "o200k_base",  # Assuming o3 uses the same encoding as o1/o200k
            "gpt-4.1": "cl100k_base"  # Fallback encoding
        }

        encoding_name = model_encodings.get(self.model_type, "cl100k_base")  # Default fallback
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(prompt))
        except Exception as e:
            print(f"Warning: Could not count tokens for model {self.model_type}: {e}")
            return len(prompt) // 4
        

    def _get_all_accumulated_mismatches(self) -> List[Dict]:
        """Get all mismatches from all iterations flattened."""
        all_mismatches = []
        for iteration_mismatches in self.iteration_mismatches:
            all_mismatches.extend(iteration_mismatches['mismatches'])
        return all_mismatches
    
    def _find_new_mismatches(self, current_mismatches: Dict) -> Dict:
        """Find mismatches that don't exist in previous iterations."""
        previous_mismatches = self._get_all_accumulated_mismatches()
        
        new_file_mismatches = {}
        
        for filename, (mismatches, ocr_data) in current_mismatches.items():
            new_mismatches = []
            for mismatch in mismatches:
                # Check if this mismatch is new
                is_new = True
                for prev_mismatch in previous_mismatches:
                    if (mismatch.get('name') == prev_mismatch.get('name') and
                        mismatch.get('column') == prev_mismatch.get('column') and
                        str(mismatch.get('raw_value')) == str(prev_mismatch.get('raw_value')) and
                        str(mismatch.get('true_value')) == str(prev_mismatch.get('true_value'))):
                        is_new = False
                        break
                
                if is_new:
                    new_mismatches.append(mismatch)
            
            if new_mismatches:
                new_file_mismatches[filename] = (new_mismatches, ocr_data)
        
        return new_file_mismatches
    
    def _update_error_counts(self, new_mismatches: Dict) -> None:
        """Update document and field error counts for prioritization."""
        for filename, (mismatches, _) in new_mismatches.items():
            self.document_error_counts[filename] += len(mismatches)
            for mismatch in mismatches:
                field = mismatch.get('column')
                if field:
                    self.field_error_counts[field] += 1

    def _prioritize_errors_for_template(self) -> List[Dict]:
        """
        Prioritize errors for inclusion in the template based on:
        1. Fields with more errors (prioritized)
        2. Documents with more errors (prioritized)
        3. Recency (more recent iterations prioritized)
        """
        all_errors = []
        
        # Collect all errors with metadata
        for iter_data in self.iteration_error_analyses:
            iteration = iter_data['iteration']
            analysis = iter_data['analysis']

            for field, errors in analysis.items():
                if field in self.fields:
                    field_error_count = self.field_error_counts.get(field, 0)
                    
                    for error in errors:
                        error_filename = error.get('filename')
                        doc_error_count = self.document_error_counts.get(error_filename, 0) if error_filename else 0
                        
                        all_errors.append({
                            'field': field,
                            'error': error,
                            'field_error_count': field_error_count,
                            'doc_error_count': doc_error_count,
                            'iteration': iteration,
                        })
        
        # Sort by priority: field errors (desc), document errors (desc), recency (desc)
        all_errors.sort(key=lambda x: (
            -x['field_error_count'],  # More field errors = higher priority
            -x['doc_error_count'],    # More document errors = higher priority
            -x['iteration']           # More recent = higher priority
        ))
        
        return all_errors
    
    def _select_exemplars_with_limits(self, prioritized_errors: List[Dict]) -> List[Dict]:
        """
        Select exemplars with limits on total count and per-field count.
        """
        selected_errors = []
        field_counts = defaultdict(int)
        
        for error_data in prioritized_errors:
            field = error_data['field']
            
            # Check if we've reached global limit
            if len(selected_errors) >= 12:
                break
                
            # Check if we've reached per-field limit
            if field_counts[field] >= 3:
                continue
                
            selected_errors.append(error_data)
            field_counts[field] += 1
        
        return selected_errors
    
    def add_iteration_mismatches(self, new_mismatches: Dict) -> None:
        """
        Add mismatches from a new iteration and update error analysis incrementally.
        
        Args:
            new_mismatches (List[Dict]): Mismatches from the current iteration
        """
        if not new_mismatches:
            print("No mismatches to add for this iteration")
            return
        
        # Find only the truly new mismatches (not seen in previous iterations)
        unique_new_mismatches = self._find_new_mismatches(new_mismatches)
        print(f"Found {len(unique_new_mismatches)} new mismatches")
        
        if not unique_new_mismatches:
            print("No new unique mismatches found - all were seen in previous iterations")
            return
        
        self._update_error_counts(unique_new_mismatches)
        
        # Add to iteration tracking
        all_new_mismatches = []
        ocr_context = {}
        
        for filename, (mismatches, ocr_data) in unique_new_mismatches.items():
            all_new_mismatches.extend(mismatches)
            ocr_context[filename] = ocr_data
        
        # Add to iteration tracking with OCR context
        self.iteration_mismatches.append({
            'mismatches': all_new_mismatches,
            'ocr_context': ocr_context,
            'file_mismatches': unique_new_mismatches
        })
        self.current_iteration += 1
        
        print(f"Iteration {self.current_iteration}: Added {len(unique_new_mismatches)} new mismatches")
        print(f"Total mismatches across all iterations: {len(self._get_all_accumulated_mismatches())}")
        
        # Generate error analysis for ONLY the new mismatches

        new_error_analysis = {}
        with ThreadPoolExecutor(max_workers=5) as executor:  # Limit concurrent API calls
            # Submit all tasks
            future_to_file = {
                executor.submit(self.get_error_analysis, mismatches, ocr_data, filename): filename
                for filename, (mismatches, ocr_data) in unique_new_mismatches.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    response = future.result(timeout=30)  # 30 second timeout per call
                    if response and 'response' in response:
                        for field, errors in response['response'].items():
                            new_error_analysis[field] = new_error_analysis.get(field, []) + errors
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")

        # Store the error analysis for this iteration
        self.iteration_error_analyses.append({
            'iteration': self.current_iteration,
            'analysis': new_error_analysis,
            'mismatch_count': len(unique_new_mismatches)
        })

        self._rebuild_error_analysis_section()  # Update the error analysis section

    def _rebuild_error_analysis_section(self) -> None:
        """Rebuild the complete error analysis section from all iterations."""
        if not self.iteration_error_analyses:
            self.base_template_parts['error_analysis_section'] = ""
            return
        
        prioritized_errors = self._prioritize_errors_for_template()
        selected_errors = self._select_exemplars_with_limits(prioritized_errors)



        error_sections = []
        
        # Add main header
        error_sections.append(self.base_template_parts['error_header'])
        
        # Process each iteration's error analysis
        for error_data in selected_errors:
            field = error_data['field']
            error = error_data['error']
            
            error_sections.append(
                self.base_template_parts['field_error'].format(
                    field=field,
                    desc=self.fields[field]['desc'],
                    pred=error['actual'],
                    reason=error['reason'],
                    true=error['expected']
                )
            )
                        
        # Update the error analysis section
        self.base_template_parts['error_analysis_section'] = ''.join(error_sections)
        
        total_iterations = len(self.iteration_error_analyses)
        total_mismatches = len(self._get_all_accumulated_mismatches())
        print(f"Rebuilt error analysis section with {total_iterations} iterations, {total_mismatches} total mismatches")

    def get_error_analysis(self, mismatches: list, ocr_data: str, filename: str) -> dict:
        llm = ChatOpenAI(model=self.model_type, temperature=0)
        llm = llm.with_structured_output(ErrorAnalysis)

        prompt = """
    You are an expert in document field extraction error analysis. 
    Your job is to take a list of mismatches between 'expected' vs. 'predicted' values, 
    With given field information and OCR context, lets think step-by-step

    1. Root Cause Analysis: Identify why the extraction model failed
    2. Self-Reflection: Generate insights about what the model might have "thought" during extraction
    3. Evidence Mapping: Connect OCR artifacts to extraction decisions
    4. Pattern Recognition: Identify systematic vs. isolated errors

    RESPONSE FORMAT:
    Return a dictionary where:
    - The 'response' key contains a dictionary
    - Each field name maps to a LIST of error objects
    - Each error object has:
    - expected: ground truth value
    - actual: model predicted value
    - reason: a concise reasoning or explanation of why the model failed after self-reflection
    - filename: the filename of the document (use: {filename})

    INPUT DATA:
    Fields:
    {fields}

    filename:
    {filename}

    OCR Context:
    {ocr_context}

    Mismatches:
    {mismatches} 
        """

        response = llm.invoke(prompt.format(fields=self.fields, ocr_context=ocr_data, mismatches=mismatches, filename=filename))
        return response
        
    def structured_model(self, fields: dict) -> Type:
        output_format = create_model(
            "DynamicFieldOutputs",
            **{key: Optional[TypeMapping.string_to_type(field["type"])] for key, field in fields.items()}

        )
        return output_format

    def output_filling(self, model_cls: dict, response: BaseModel) -> tuple[Type, List[str]]:
        missing = set()
        filled_model = response.model_dump()

        for name in model_cls:
            if model_cls[name]:
                continue
            elif filled_model[name] is None or filled_model[name] == "":
                missing.add(name)
            else:
                model_cls[name] = filled_model[name]
        
        return model_cls, missing

    def get_response(self, fields: dict, kvs: dict = None, ocr: str = None, coordinates: list = None):
        llm = ChatOpenAI(model=self.model_type, temperature=0)
        output_format = self.structured_model(fields)

        llm = llm.with_structured_output(output_format)
        prompt, token_count = self.generate_prompt(fields=fields, kvs=kvs, ocr=ocr, coordinates=coordinates)
        
        print(f"Prompt token count: {token_count}")
        
        response = llm.invoke(prompt)

        return response

    def fv_extract(self, kvs: dict[str, list[str]] = None, ocr: str = None, coordinates: list = None) -> dict[str, str]:
        '''
        Extract values from a key-value dictionary based on field descriptions and types.
        
        Parameters:
        fields (dict): A dictionary containing field descriptions, types, and examples.
        kvs (dict): A dictionary containing key-value pairs to extract values from.
        
        Returns:
        dict: A dictionary containing the extracted values for each field.
        '''
        
        # Track the number of attempts and Initialize results and incomplete fields
        attempts = 0
        results = {k: None for k in self.fields.keys()}
        incomplete_fields = {k: self.fields[k] for k in self.fields.keys()}
        
        # Iterate until all fields are filled or max attempts reached
        while attempts < self.max_attempts:
            try:
                print('Attempt #', attempts + 1)
                
                # Query the OpenAI API with the prompt
                response = self.get_response(fields=incomplete_fields, kvs=kvs, ocr=ocr, coordinates=coordinates)
                results, missing = self.output_filling(results, response)

                # Check for incomplete fields
                incomplete_fields = {
                    k: incomplete_fields[k]
                    for k in missing
                }
                
                # If all fields are filled, break the loop
                if len(incomplete_fields) == 0:
                    break
                
                attempts += 1
                
            except json.JSONDecodeError:
                # If JSON decoding fails, return the raw result
                print(f"Failed to decode JSON: {response}")
            
        return results
