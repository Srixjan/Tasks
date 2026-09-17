from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def build_feature_pipeline(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    col_trasnform = ColumnTransformer(
        transformers = [
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    return col_trasnform

# build_feature_pipeline:
# this function primary task is to transform categorical features aka state columns into OneHotEncoding and ignoring numeric_features.