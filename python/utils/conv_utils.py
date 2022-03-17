
#tensorflow 참고

def normalize_tuple(value, rank):
    if isinstance(value, int):
        value_tuple = (value, ) * rank
    elif isinstance(value, (tuple, list)):
        value_tuple = tuple(value)
        if len(value) != rank:
            raise ValueError("입력 인자의 형태가 올바르지 않습니다. value : {}, rank : {}".format(value, rank))
        else:
            for single_value in value_tuple:
                if not isinstance(single_value, int):
                    raise ValueError("입력 인자의 타입이 올바르지 않습니다. value : {}, single_value : {}".format(value_tuple, single_value))

    return value_tuple

def normalize_padding(value):
    if not isinstance(value, str):
        raise ValueError("입력 인자 패딩 타입 올바르지 않습니다. value : {}, type : {}".format(value, type(value)))
    padding = value.lower()
    if padding not in ('valid', 'same', 'causal'):
        raise ValueError("입력 패딩 종류가 valid, same, causal 중 하나가 아닙니다. value : {}".format(value))
    return padding

