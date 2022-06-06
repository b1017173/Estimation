class Attribute:
    def __init__(self, name:str, type:str, properties:list = []):
        self.name = name
        self.type = type
        self.properties = properties
        self.is_activate:bool

    def set_activate(self, is_activate:bool):
        self.is_activate = is_activate
