import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "department": ["Engineering", "Sales", "HR", "Marketing",
                    "Engineering", "Sales", "HR", "Engineering"],
    "salary": [85000, 62000, 58000, 65000,
               91000, 59000, 60000, 88000]
}

df = pd.DataFrame(data)
def encode_categorical(df: pd.DataFrame, column: str, drop_first: bool=True) -> pd.DataFrame:
    return pd.get_dummies(df, prefix=column, drop_first=drop_first)

dummies_false = encode_categorical(df, column="department", drop_first=False)
print("drop_first=False:\n", dummies_false, "\n")

dummies_true = encode_categorical(df, column="department", drop_first=True)
print("drop_first=True:\n", dummies_true, "\n")