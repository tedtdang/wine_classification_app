import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler
)
from typeguard import typechecked

np.random.seed(11)
set_config(display="diagram")


class Processor:
    @typechecked
    def __init__(self, data_path: str, target_column: str):
        df = pd.read_csv(data_path)
        self.target_column = target_column
        self.df = df.dropna(subset=self.target_column)
        self.X = self.df.drop(columns=self.target_column)
        self.y = self.df[self.target_column]
        self.num_cols = self.X.select_dtypes(include="number").columns
        self.cat_cols = list(set(self.X.columns) - set(self.num_cols))

    def process(self):
        union = FeatureUnion(
            transformer_list=[
                ("pca", PCA(n_components=1)),
                ("svd", TruncatedSVD(n_components=2)),
            ]
        )

        # Get numeric and categorical columns

        num_pipe = Pipeline(
            [
                ("Impute", KNNImputer()),
                ("Scale", StandardScaler()),
                # ('reduce_dimensionality', union)
            ]
        )
        cat_pipe = Pipeline(
            [
                ('encode', CatBoostEncoder(cols=self.cat_cols)),
                ("Impute", KNNImputer()),
            ]
        )

        # Define the pipeline with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", cat_pipe, self.cat_cols),
                ("numeric", num_pipe, self.num_cols),
            ],
            remainder="passthrough",
            verbose_feature_names_out=True,
        )

        processor = Pipeline(
            [
                ("preprocessing", preprocessor),
                # ("feature_engineering", feature_engineering),
            ]
        )
        self.X_transformed = processor.fit_transform(self.X)
        model = RandomForestClassifier()
        model.fit(self.X_transformed, self.y)
        return processor, model
