import lightgbm as lgb
from base_model import BaseModel
from metrics import opt_smape1p


class LGBClassModel1(BaseModel):
    """LightGBM classification model."""

    def __init__(self, params, features):
        self.params = params
        self.features = features

    def fit(self, df_train):
        if self.features is None:
            self.features = [col for col in df_train.columns if col.startswith("v_")]
        lgb_train = lgb.Dataset(df_train[self.features], df_train["target"])
        params0 = {k: v for k, v in self.params.items() if k not in ["n_estimators"]}
        self.m_gbm = lgb.train(
            params0, lgb_train, num_boost_round=self.params["n_estimators"]
        )
        return self

    def predict_proba(self, df_valid):
        return self.m_gbm.predict(df_valid[self.features])

    def predict(self, df_valid):
        return opt_smape1p(self.predict_proba(df_valid))
