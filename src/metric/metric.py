from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self):
        pass

    """
    定义一个 Metric 接口，所有指标类必须实现 __call__ 方法。
    """
    @abstractmethod
    def __call__(self, input_paths, keys):
        """
        所有指标类必须实现的方法。
        :param input_paths: 输入路径列表
        :param keys: 用于计算指标的关键字列表
        :return: 指标分数
        """
        pass