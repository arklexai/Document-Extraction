class Example:
    def __init__(self, base=None, fields=None, context=None, mode=None, output=None):
        # Internal storage
        self._store = {}
        self._demos = []
        self._input_keys = {'fields', 'context', 'mode'}

        # Initialize from a base Example if provided
        if base and isinstance(base, type(self)):
            self._store = base._store.copy()
            self._input_keys = base._input_keys.copy()

        # Initialize from a dict if provided
        elif base and isinstance(base, dict):
            self._store = base.copy()

        if fields is not None:
            self._store['fields'] = fields
        if context is not None:
            self._store['context'] = context
        if mode is not None:
            self._store['mode'] = mode
        if output is not None:
            self._store['output'] = output

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError
        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key.startswith("_") or key in dir(self.__class__):
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __contains__(self, key):
        return key in self._store

    def keys(self):
        return self._store.keys()
    
    def values(self):
        return self._store.values()
    
    def get(self, key, default=None):
        return self._store.get(key, default)

    def inputs(self):
        if self._input_keys is None:
            raise ValueError("Inputs have not been set for this example.")

        d = {key: self._store[key] for key in self._store if key in self._input_keys}
        new_instance = type(self)(base=d)
        new_instance._input_keys = self._input_keys
        return new_instance

    def labels(self):
        input_keys = self.inputs().keys()
        d = {key: self._store[key] for key in self._store if key not in input_keys}
        return type(self)(d)

    def copy(self, **kwargs):
        return type(self)(base=self, **kwargs)

    def without(self, *keys):
        copied = self.copy()
        for key in keys:
            if key in copied._store:
                del copied._store[key]
        return copied

    def to_dict(self):
        return self._store.copy()