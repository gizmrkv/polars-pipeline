from typing import List


class NotFittedError(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"{name} is not fitted")


class LazyFrameNotSupportedError(Exception):
    def __init__(self, name: str, method: str):
        self.name = name
        self.method = method
        super().__init__(f"LazyFrame not supported: {name}.{method}")


class ColumnsMismatchError(Exception):
    def __init__(self, name: str, columns: List[str], fitted_columns: List[str]):
        self.name = name
        self.columns = columns
        self.fitted_columns = fitted_columns
        msg = f"Columns of X do not match the columns of the fitted {name}.\n"
        msg += f"Columns of X: {columns}\n"
        msg += f"Columns of the fitted {name}: {fitted_columns}"
        super().__init__(msg)
