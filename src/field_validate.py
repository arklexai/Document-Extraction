
def validate(field_type: str, value: str, required: bool=True) -> bool:
    '''
    Validate the field value based on its type.
    
    Parameters:
    field_type (str): The type of the field ('int', 'float', 'string').
    value (any): The value to validate.
    required (bool): Whether the field is required or not. Default is True.
    
    Returns:
    bool: True if the value is valid, False otherwise.
    '''
    
    # Check if the field itself is not found
    if value is None:
        return False
    
    # Check if the field is required and empty
    if str(value).strip() == '':
        return not required
    
    # Check if the field is required and not empty
    match field_type:
        case 'int':
            try:
                int(value)
                return True
            except ValueError:
                return False
            
        case 'float':
            try:
                float(value)
                return True
            except ValueError:
                return False
            
        case 'string':
            try:
                str(value)
                return True
            except ValueError:
                return False
                
        case _:
            raise ValueError(f"Unknown field type: {field_type}")
        