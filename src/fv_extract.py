from dotenv import load_dotenv
from transformers import AutoTokenizer
import tiktoken

from field_validate import validate
load_dotenv()

import json
from pydantic import BaseModel, create_model
from typing import Optional, Dict, List, Type

from model import get_model

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
    
class FVExtraction:
    def __init__(self, fields: dict[str, dict], model_type: str="gpt-4o-mini", max_attempts: int = 3):
        self.llm = get_model(model_type)
        self.fields = fields
        self.model_type = model_type
        self.max_attempts = max_attempts
        self.base_template_parts = self._initialize_base_template()

        self.meta_prompt = None
        self.positive_exemplars = []
        

    def _initialize_base_template(self) -> Dict[str, str]:
        """Initialize the base prompt template parts."""
        return {
            'meta_prompt_instructions': """
            [Meta-prompt Instructions]:
            {meta_instructions}
            ====End of Meta-prompt Instructions====
            """,
            'base': """
            Extract value from a key:value dictionary list, OCR result, and/or coordinate layout given a description and type for each field in a fields dictionary.

            **CRITICAL RULE: All extracted values MUST come exclusively from the INPUT DATA section. Do not use any information from examples, previous extractions, or external knowledge.**

            Return key-value results in a JSON dictionary string, where:
            - If a field is **not found**, return `null`.
            - If a field is found but has **no value**, return an **empty string**.
            - Values must respect their declared types (e.g., float, string).
            - Output must be compatible with `json.loads()` (i.e., double-quoted strings).
            - Provide the final answer with rationale.

            {mode_instructions}
            """,
            'kvs': """
            Matching Strategy:
              1. Normalize for comparison: Convert keys to lowercase, remove extra spaces/punctuation
              2. Priority matching order:
                - Exact field identifier match (e.g., field 'a' matches key starting with 'a ')
                - Semantic similarity (e.g., "SSN" matches "social security number")
                - Keyword overlap (e.g., "wages" in field matches "wages" in key)
                - Type-based inference (e.g., 9-digit number for SSN field)
            Value Selection:
              - If value is a list, typically use the first clean entry
              - Trim whitespace from extracted values (preserve essential formatting)
              - Validate against expected type and format
            """,
            'ocr': """
            Extraction Strategy:
              1. Pattern Recognition: Look for "field_description: value" or "field_code value" patterns
              2. Proximity Analysis: Find values on the same line or immediately following field descriptions
              3. Multi-line Handling: For addresses/complex fields, capture continuation lines
              4. Format Validation: Use expected patterns (9 digits for SSN, decimal for wages, etc.)
            
            Value Extraction:
              - Extract values that appear after field identifiers
              - Handle multi-line values (especially addresses)
              - Clean extracted values but preserve essential formatting
            """,
            'spatial': """
            1. Confidence Filtering: Prioritize OCR results with confidence > 0.8
            2. Spatial Relationships:
              - Same row: y-coordinates within ±5 pixels
              - Same column: x-coordinates within ±10 pixels
              - Adjacent positioning: closest text element to field label
            3. Alignment Detection: Find horizontally aligned label-value pairs
            
            Value Selection:
              - Choose the highest confidence text element that's spatially related to the field
              - Prefer values that are properly aligned (same row/column)
              - Validate extracted values against expected type and format
            """,
            'w2_example': """
            ==Positive Examples==
            Example Fields dict:
            {{
                'a': {{'desc': 'a: Employee's Social Security Number (SSN) [9 digits]', 'type': 'string', 'example': '123456789', 'required': True}},
                '1': {{'desc': '1 Wages tips other comp', 'type': 'float', 'example': 1234.56, 'required': True}},
                'e': {{'desc': 'e: Employee's Name', 'type': 'string', 'example': 'John Smith', 'required': True}},
                'f': {{'desc': 'f: Employee's Address', 'type': 'string', 'example': '123 Main St, City, State, ZIP', 'required': True}},
                '12a': {{'desc': '12a: code', "type": "string", 'example': 'D 6,281.52', 'required': False}}
                '14': {{'desc': '14: Other', 'type': 'string', 'example': 'DCP-REG 916.18  CA-SDI 66.53', 'required': False}},
                            
            }}

            Example Key-Value dict:
            {{
                'a Employee's social security number'  : ['987654321 ', '987654321 ', '987654321 ', '987654321 ']
                '1 Wages tips other comp'  : ['52345.67 ', '52345.67 ', '52345.67 ', '52345.67 ']
                'e Employee's name, address and ZIP code'  : ['Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ', 'Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ', 'Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 ', 'Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445 '],
                '12a code see inst, for box 12'  : ['D 6,281.52', 'D 6,281.52', 'D 6,281.52', 'D 6,281.52'],
                '14 Other'  : ['DCP-REG 916.18  CA-SDI 66.53', 'DCP-REG 916.18', 'CA-SDI 66.53', 'DCP-REG 916.18  CA-SDI 66.53']
            }}

            Example OCR text result:
            '''
            a Employee's SSN  987654321
            1 Wages tips other comp  52345.67
            e Employee's name, address and ZIP code  
            Yi Ding 6709 SETZLER PARKWAY BROOKLYN PARK MN 55445
            12a code see inst, for box 12  
            D   6,281.52
            14 Other  DCP-REG 916.18  
            CA-SDI 66.53
            '''

            Example JSON output:
            '''
            {{
                'a': '987654321',
                'e': 'Yi Ding',
                'f': '6709 SETZLER PARKWAY BROOKLYN PARK MN 55445',
                '1': 52345.67,
                '12a': 'D 6,281.52',
                '14': 'DCP-REG 916.18  CA-SDI 66.53'
            }}
            '''
            ===END OF POSITIVE EXAMPLES===
            """,
            'mismatch': """
            === PREVIOUS EXTRACTION EXEMPLARIES ===
            {mismatches}
            ===END OF PREVIOUS EXTRACTION EXEMPLARIES===
            """,
            'input': """
            === INPUT DATA ===
            Input Fields dict: 
            {fields}

            Input OCR Data (key:value dict, OCR text, or OCR text with coordinates):
            {input_data}
            ===END OF INPUT DATA===
            """
        }

    def generate_prompt(self, fields: dict, kvs: Dict = None, ocr: str = None, coordinates: List = None) -> tuple[str, int]:
        # Build prompt efficiently using string templates
        prompt_parts = []

        if self.meta_prompt:
            tag = self.meta_prompt.tag
            instructions = "\n".join(self.meta_prompt.instruction)
            meta_section = self.base_template_parts['meta_prompt_instructions'].format(meta_instructions=f"<{tag}>\n{instructions}\n</{tag}>")
            prompt_parts.append(meta_section)
        
        if kvs:
            mode_instructions = self.base_template_parts['kvs']
        elif ocr:
            mode_instructions = self.base_template_parts['ocr']
        elif ocr and coordinates:
            mode_instructions = self.base_template_parts['spatial']

        prompt_parts.append(self.base_template_parts['base'].format(mode_instructions=mode_instructions))
        prompt_parts.append(self.base_template_parts['w2_example'])

        input_data = []
        if kvs:
            input_data.append(f"Input Key-Value dict:\n{kvs}")
        if ocr:
            input_data.append(f"Input OCR text result:\n{ocr}")
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
        openai_model_encodings = {
            "gpt-4o-mini": "o200k_base",
            "gpt-4o": "o200k_base", 
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-o3": "o200k_base",  # Assuming o3 uses the same encoding as o1/o200k
            "gpt-4.1": "cl100k_base"  # Fallback encoding
        }

        hf_model_tokenizers = {
            "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
            "mistralai/Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2"
        }
        
        # claude token count tbd
        try:
            if self.model_type in openai_model_encodings:
                encoding_name = openai_model_encodings[self.model_type]
                encoding = tiktoken.get_encoding(encoding_name)
                return len(encoding.encode(prompt))
            elif self.model_type in hf_model_tokenizers:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_tokenizers[self.model_type])
                return len(tokenizer.encode(prompt))
            else:
                print(f"Warning: Unknown model type {self.model_type}, using fallback estimate.")
            return len(prompt) // 4
        except Exception as e:
            print(f"Warning: Could not count tokens for model {self.model_type}: {e}")
            return len(prompt) // 4
        
    def structured_model(self, fields: dict, include_rationale: bool) -> Type:
        fields_model = create_model(
            "DynamicFieldOutputs",
            **{key: Optional[TypeMapping.string_to_type(field["type"])] 
            for key, field in fields.items()}
        )

        output_model = create_model(
            "ExtractorOutput",
            result=(fields_model, ...)
        )

        if include_rationale:
            output_model = create_model(
                "ExtractorOutput",
                result=(fields_model, ...),
                logic=(str, ...)
            )
            return output_model
        
        return output_model

    def output_filling(self, model_cls: dict, response: BaseModel) -> tuple[Type, List[str]]:
        missing = set()
        filled_model = response.model_dump()

        for name in model_cls:
            if name in filled_model:
                model_cls[name] = filled_model[name]
            
            if not validate(self.fields[name]['type'], model_cls[name], self.fields[name]['required']):
                missing.add(name)

        return model_cls, missing

    def fv_extract(self, kvs: dict[str, list[str]] = None, ocr: str = None, coordinates: list = None, include_rationale: bool = False) -> dict:
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
        results, rationale = {k: None for k in self.fields.keys()}, None
        incomplete_fields = {k: self.fields[k] for k in self.fields.keys()}
        
        # Iterate until all fields are filled or max attempts reached
        while attempts < self.max_attempts:
            try:
                print('Attempt #', attempts + 1)
                
                output_format = self.structured_model(incomplete_fields, include_rationale=include_rationale)
                prompt, token_count = self.generate_prompt(fields=incomplete_fields, kvs=kvs, ocr=ocr, coordinates=coordinates)
                response = self.llm.get_response(prompt, output_format=output_format)
                results, missing = self.output_filling(results, response.result if include_rationale else response)

                if include_rationale and hasattr(response, 'logic'):
                    rationale = response.logic

                incomplete_fields = {k: incomplete_fields[k] for k in missing}

                
                # If all fields are filled, break the loop
                if len(incomplete_fields) == 0:
                    break
                
                attempts += 1
                
            except json.JSONDecodeError:
                # If JSON decoding fails, return the raw result
                print(f"Failed to decode JSON: {response}")
            
        return {
            'result': results,
            'rationale': rationale
        }