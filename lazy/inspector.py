import inspect


def get_class_params(cls, func_name='__init__'):
    """Get the `__init__` parameters of the class."""
    init_method = getattr(cls, func_name, None)
    if not init_method:
        return []

    signature = inspect.signature(init_method)
    return list(signature.parameters.keys())


def filter_func_params(param_dict, cls, func_name='__init__'):
    """
    Filter param_dict to only include parameters accepted by the class method.

    Args:
        cls: Class or class instance.
        func_name: Function name as string.
        param_dict: Dictionary of parameters to filter.

    Returns:
        A filtered dictionary containing only parameters accepted by the function.
    """
    valid_params = get_class_params(cls, func_name)
    return {k: v for k, v in param_dict.items() if k in valid_params}


def check_init_parameter(cls, param_name):
    """Check whether the `__init__` method of the class requires the parameter `param_name` to be specified."""
    init_method = getattr(cls, "__init__", None)
    if not init_method:
        return False

    signature = inspect.signature(init_method)
    return param_name in signature.parameters


def call_with_required(func, kwargs):
    """
    调用函数，只使用该函数所需要的关键词参数。

    :param func: 要调用的函数
    :param kwargs: 提供的关键字参数字典
    :return: 函数的返回值
    """
    signature = inspect.signature(func)
    required_params = [k for k in signature.parameters.keys()]

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in required_params}

    return func(**filtered_kwargs)


if __name__ == '__main__':
    def example_function(a, b, c=None):
        return a + b + (c if c is not None else 0)


    # 测试
    result = call_with_required(example_function, {'a': 1, 'b': 2, 'c': 3, 'd': 3})
    print(result)
