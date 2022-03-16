

def parse_activation(identifier):
    return globals().get(identifier)

def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, str):
        return parse_activation(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise TypeError("")