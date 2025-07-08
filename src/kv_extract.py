import boto3
import sys
import re
import pymupdf
import json
from collections import defaultdict
from multiprocessing import Pool

class KVExtractDocument:
    '''
    KVExtractDocument class to extract key-value pairs from a PDF document using AWS Textract.
    
    Parameters:
    filepath (str): The path to the PDF file.
    page (int, optional): The specific page number to extract. If None, all pages are extracted.
    max_processes (int): The maximum number of processes to use for parallel processing.
    
    Attributes:
    filepath (str): The path to the PDF file.
    document (pymupdf.Document): The PDF document object.
    pages (list): A list of KVExtractPage objects representing the extracted pages.
    
    Methods:
    __init__(self, filepath, page=None, max_processes=4):
        Initializes the KVExtractDocument object.
    '''
    def __init__(self, filepath, page=None, max_processes=4):
        '''
        Initializes the KVExtractDocument object. Extracts key-value pairs from the PDF document using AWS Textract.
        
        Parameters:
        filepath (str): The path to the PDF file.
        page (int, optional): The specific page number to extract. If None, all pages are extracted.
        max_processes (int): The maximum number of processes to use for parallel processing.
        '''
        self.filepath = filepath
        self.document = pymupdf.open(filepath) if filepath.endswith('.pdf') else None
        self.pages = []

        if filepath.endswith('.jpg') or filepath.endswith('.png'):
            with open(filepath, 'rb') as file:
                img = file.read()
                bytes = [bytearray(img)]
                print('Image loaded', filepath)
        
        elif filepath.endswith('.pdf'):
            if page is None:
                bytes = [page.get_pixmap().tobytes() for page in self.document]
            else:
                bytes = [self.document[page].get_pixmap().tobytes()]
        
            
        # Use multiprocessing to process the pages in parallel
        with Pool(processes=max_processes) as pool:
            # Process each page in parallel
            self.pages = pool.map(KVExtractPage, bytes)
        # self.pages = [KVExtractPage(page) for page in self.document]
        
class KVExtractPage:
    '''
    KVExtractPage class to extract key-value pairs from a single page of a PDF document using AWS Textract.
    
    Parameters:
    bytes (bytes): The bytes of the PDF page.
    
    Attributes:
    kvs (defaultdict): A dictionary mapping keys to lists of values.
    key_map (dict): A dictionary mapping block IDs to key blocks.
    value_map (dict): A dictionary mapping block IDs to value blocks.
    block_map (dict): A dictionary mapping block IDs to all blocks.
    
    Methods:
    __init__(self, bytes):
        Initializes the KVExtractPage object.
        
    get_kv_relationship(self):
        Extracts the key-value relationships from the blocks.
        
    find_value_block(self, key_block):
        Finds the corresponding value block for a given key block.
        
    get_text(self, result):
        Extracts the text from a block.
        
    print_kvs(self):
        Prints the key-value pairs.
        
    search_value(self, search_key):
        Searches for a value associated with a given key.
    '''
    def __init__(self, bytes):
        client = boto3.client('textract', region_name='us-east-1')
        response = client.analyze_document(Document={'Bytes': bytes}, FeatureTypes=['FORMS'])

        # Get the text blocks
        blocks = response['Blocks']

        # get key and value maps
        self.key_map = {}
        self.value_map = {}
        self.block_map = {}
        for block in blocks:
            block_id = block['Id']
            self.block_map[block_id] = block
            if block['BlockType'] == "KEY_VALUE_SET":
                if 'KEY' in block['EntityTypes']:
                    self.key_map[block_id] = block
                else:
                    self.value_map[block_id] = block

        self.kvs = self.get_kv_relationship()
        self.position = self.get_coordinates()

    def get_kv_relationship(self) -> dict[str, list[str]]:
        '''
        Extracts the key-value relationships from the blocks.
        
        Returns:
        kvs (defaultdict): A dictionary mapping keys to lists of values.
        '''
        kvs = defaultdict(list)
        for block_id, key_block in self.key_map.items():
            value_block = self.find_value_block(key_block)
            key = self.get_text(key_block)
            val = self.get_text(value_block)
            kvs[key].append(val)
        return kvs
    
    def get_coordinates(self) -> dict[str, list[str]]:
        '''
        Extracts the coordinates for each key-value pair.
        
        Returns:
        dict: A dictionary mapping keys to lists of dictionaries containing value and coordinates
        '''
        coordinates = defaultdict(list)
        for block_id, key_block in self.key_map.items():
            value_block = self.find_value_block(key_block)
            key = self.get_text(key_block)
            
            # Get key coordinates
            key_bbox = key_block['Geometry']['BoundingBox']
            key_coords = {
                'x': key_bbox['Left'],
                'y': key_bbox['Top'],
                'width': key_bbox['Width'],
                'height': key_bbox['Height']
            }
            
            # Get value coordinates
            value_bbox = value_block['Geometry']['BoundingBox']
            value_coords = {
                'x': value_bbox['Left'],
                'y': value_bbox['Top'],
                'width': value_bbox['Width'],
                'height': value_bbox['Height']
            }
            
            # Get the actual value text
            val = self.get_text(value_block)
            
            coordinates[key].append({
                'value': val,
                'key_coordinates': key_coords,
                'value_coordinates': value_coords
            })
            
        return coordinates


    def find_value_block(self, key_block: dict) -> dict:
        for relationship in key_block['Relationships']:
            if relationship['Type'] == 'VALUE':
                for value_id in relationship['Ids']:
                    value_block = self.value_map[value_id]
        return value_block


    def get_text(self, result):
        text = ''
        if 'Relationships' in result:
            for relationship in result['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        word = self.block_map[child_id]
                        if word['BlockType'] == 'WORD':
                            text += word['Text'] + ' '
                        if word['BlockType'] == 'SELECTION_ELEMENT':
                            if word['SelectionStatus'] == 'SELECTED':
                                text += 'X '

        return text

    def print_kvs(self):
        for key, value in self.kvs.items():
            print(key, ":", value)

    
    def print_coordinates(self):
        for key, data in self.position.items():
            print(f"\nKey: {key}")
            for item in data:
                print(f"Value: {item['value']}")
                print(f"Key coordinates: {item['key_coordinates']}")
                print(f"Value coordinates: {item['value_coordinates']}")


    def search_value(self, search_key):
        for key, value in self.kvs.items():
            if re.search(search_key, key, re.IGNORECASE):
                return value
    

    def search_coordinates(self, search_key):
        for key, data in self.position.items():
            if re.search(search_key, key, re.IGNORECASE):
                return data

def main(file_name: str):
    # key_map, value_map, block_map = get_kv_map(file_name)

    # # Get Key Value relationship
    # kvs = get_kv_relationship(key_map, value_map, block_map)
    kv = KVExtractDocument(file_name)
    print("\n\n== FOUND KEY : VALUE pairs ===\n")
    kv.print_kvs()

    # Start searching a key value
    while input('\n Do you want to search a value for a key? (enter "n" for exit) ') != 'n':
        search_key = input('\n Enter a search key:')
        print('The value is:', kv.search_value(search_key))

if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)