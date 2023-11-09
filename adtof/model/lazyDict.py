from collections.abc import Mapping


class LazyDict(Mapping):
    def __init__(self, func, store=True):
        """
        Implement a dictionary instanciated with a function to build the elements.
        The function is called with the key of the element as a parameter, only when the element is accessed.
        The result is then stored.

        usage:
        lazy = LazyDict(lambda key: expensive(key))

        lazy['A'] -> compute, store and return the value.

        """
        self.store = store
        self.func = func
        self.valuesStorage = dict()

    def __getitem__(self, key):
        if self.store:
            if key not in self.valuesStorage:
                self.valuesStorage[key] = self.func(key)
            return self.valuesStorage[key]
        else:
            return self.func(key)
        
    def getWithoutSaving(self, key):        
        return self.valuesStorage[key] if key in self.valuesStorage else self.func(key)

    def reset(self):
        self.valuesStorage = dict()

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)
