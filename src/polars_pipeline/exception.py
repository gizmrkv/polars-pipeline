class NotFittedError(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"{name} is not fitted")


class LazyFrameNotSupportedError(Exception):
    def __init__(self, name: str, method: str):
        self.name = name
        self.method = method
        super().__init__(f"LazyFrame not supported: {name}.{method}")
