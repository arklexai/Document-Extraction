from abc import ABC, abstractmethod
import base64
import io
import logging
import os
import sys
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import boto3
from pdf2image import convert_from_path
from mistralai import Mistral


logger = logging.getLogger(__name__)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

@dataclass
class OCRResult:
    """Data class to store OCR result for a single text block"""
    value: str
    type: Optional[str] = None  # text, number, figure, table, etc.
    geometry: Optional[List[Dict[str, float]]] = None  # coordinates of the text block
    confidence: Optional[float] = None  # confidence score of the OCR result

class OCRMethod(ABC):
    """Abstract base class for OCR methods"""
    @abstractmethod
    def process(self, filepath: str) -> List[OCRResult]:
        """Process the file and return OCR results"""
        pass

class TextractOCR(OCRMethod):
    """AWS Textract implementation"""
    def process(self, filepath: str) -> List[OCRResult]:
        print("Processing with Textract")
        client = boto3.client('textract', region_name='us-east-1')

        file_path = Path(filepath)
        ext = file_path.suffix.lstrip('.')

        if ext not in  ['jpg', 'jpeg', 'png', 'pdf']:
            raise ValueError(f"Unsupported file format: {ext}")

        # currently pdf needs to be converted to image -> is this the best way to do this?
        if ext == 'pdf':
            images = convert_from_path(file_path)

            results = []
            for page_num, image in enumerate(images):
                print(f"Processing page {page_num + 1}")
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                file_bytes = img_byte_arr.getvalue()

                response = client.detect_document_text(Document={'Bytes': file_bytes})

                page_words = []
                for block in response['Blocks']:
                    if block['BlockType'] == 'WORD':
                        result = OCRResult(
                            value=block['Text'],
                            type='text',
                            geometry=[{
                                'x': block['Geometry']['BoundingBox']['Left'],
                                'y': block['Geometry']['BoundingBox']['Top'],
                                'width': block['Geometry']['BoundingBox']['Width'],
                                'height': block['Geometry']['BoundingBox']['Height']
                            }],
                            confidence=block.get('Confidence', 0.0)
                        )
                        page_words.append(result)
                results.append(page_words)
        else:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            response = client.detect_document_text(Document={'Bytes': file_bytes})

            results, page_words = [], []
            for block in response['Blocks']:
                if block['BlockType'] == 'WORD':
                    result = OCRResult(
                        value=block['Text'],
                        type='text',
                        geometry=[{
                            'x': block['Geometry']['BoundingBox']['Left'],
                            'y': block['Geometry']['BoundingBox']['Top'],
                            'width': block['Geometry']['BoundingBox']['Width'],
                            'height': block['Geometry']['BoundingBox']['Height']
                        }],
                        confidence=block.get('Confidence', 0.0)
                    )
                    page_words.append(result)
            results.append(page_words)
                  
        return results

class MistralOCR(OCRMethod):
    """Mistral OCR implementation"""
    def process(self, filepath: str) -> List[OCRResult]:
        print("Processing with Mistral")

        client = Mistral(api_key=MISTRAL_API_KEY)

        file_path = Path(filepath)
        file_name = file_path.name
        ext = file_path.suffix.lstrip('.')
        
        if ext not in  ['jpg', 'jpeg', 'png', 'pdf']:
            raise ValueError(f"Unsupported file format: {ext}")
        
        if ext == 'pdf':
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": file_name,
                    "content": open(file_path, "rb"),
                },
                purpose="ocr"
            )
            signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                }
            )
        else:
            base64_image = self.encode_image(filepath)
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/{ext};base64,{base64_image}" 
                }
            )

        results = []
        for page in ocr_response.pages:
            result = OCRResult(
                value=page.markdown,
                type='text',
                geometry=[{
                    'x': 0,
                    'y': 0,
                    'width': page.dimensions.width,
                    'height': page.dimensions.height,
                }],
                confidence=None
            )
            results.append(result)
            
        return results

    def encode_image(self, image_path: str):
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Error: The file {image_path} was not found.")
            return None
        except Exception as e:  # Added general exception handling
            logger.error(f"Error: {e}")
            return None

class OCR:
    """
    Main OCR class that handles different OCR methods and file processing
    """
    def __init__(self, filepath: str, method: OCRMethod):
        """
        Initialize OCR with file and OCR method
        
        Args:
            file: Path to file or file bytes
            method: OCRMethod implementation (TextractOCR, MistralOCR, etc.)
        """
        self.filepath = filepath
        self.method = method
        
    def process(self) -> List[OCRResult]:
        """
        Process the file using the specified OCR method
        
        Returns:
            List of OCRResult objects containing the extracted text and metadata
        """
        return self.method.process(self.filepath)
    
    def get_text_blocks(self) -> List[Dict[str, Any]]:
        """
        Get text blocks with their geometry and type information
        
        Returns:
            List of dictionaries containing text blocks and their metadata
        """
        results = self.process()
        text_blocks = []

        
        # Textract OCR saved in List[List[OCRResult]] and Mistral OCR saved in List[OCRResult]
        # Textract OCR saves list of words per page. Each OCRResult is a word with geometric location
        # Mistral OCR saves list of OCRResult. Each OCRResult is a text from page with page dimensions
        if results and isinstance(results[0], list):
            for page_num, page_words in enumerate(results):
                page_text = ' '.join(word.value for word in page_words)
                page_geometry = [word.geometry[0] for word in page_words]
                page_confidence = sum(word.confidence for word in page_words) / len(page_words) if page_words else 0.0
                
                text_blocks.append({
                    'page': page_num + 1,
                    'value': page_text,
                    'type': 'text',
                    'geometry': page_geometry,
                    'confidence': page_confidence,
                    'words': [
                        {
                            'value': word.value,
                            'type': word.type,
                            'geometry': word.geometry,
                            'confidence': word.confidence
                        } for word in page_words
                    ]
                })
        else:
            for page_num, result in enumerate(results):
                text_blocks.append({
                    'page': page_num + 1,
                    'value': result.value,
                    'type': result.type,
                    'geometry': result.geometry,
                    'confidence': result.confidence,
                    'words': [{
                        'value': result.value,
                        'type': result.type,
                        'geometry': result.geometry,
                        'confidence': result.confidence
                    }]
                })
        
        return text_blocks
    
    def get_plain_text(self) -> str:
        """
        Get plain text from OCR results
        
        Returns:
            Concatenated text from all OCR results
        """
        results = self.process()
        page_texts = []
        
        if results and isinstance(results[0], list):
            for page_words in results:
                page_text = ' '.join(word.value for word in page_words)
                page_texts.append(page_text)
        else:
            for result in results:
                page_texts.append(result.value)

        return '\n\n'.join(page_texts)

def main(file_name: str):
    # Create OCR method instance
    textract = TextractOCR()
    
    # Create OCR instance with file and method
    ocr = OCR(file_name, textract)
    
    # Get all text blocks with metadata
    text_blocks = ocr.get_text_blocks()
    
    # Get plain text
    text = ocr.get_plain_text()


if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)