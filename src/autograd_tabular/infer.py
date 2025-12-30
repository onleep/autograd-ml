import hydra
import pandas as pd
from dvc.repo import Repo
from mlflow import catboost
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer


def embedding(data: pd.DataFrame, column: str) -> pd.DataFrame:
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    texts = data[column].astype(str).fillna('').tolist()
    embeds = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeds = pd.DataFrame(
        embeds,
        index=data.index,
        columns=[f'desc_{i}' for i in range(embeds.shape[1])],
    )
    return pd.concat([data.drop(columns=['description']), embeds], axis=1)


def preprocess(data: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    x = data.drop(columns=cfg.data.drop_cols)
    int_cols = [c for c in x.columns if x[c].dtype == 'int64']
    x[int_cols] = x[int_cols].astype(float)
    cat_cols = [c for c in x.columns if x[c].dtype == 'object']
    x[cat_cols] = x[cat_cols].astype(str)
    return x


@hydra.main(config_path='../../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    repo = Repo()
    repo.pull(targets=['download'])
    dataset_path = str(repo.index.stages[0].outs[0])
    data = pd.read_csv(dataset_path)
    embeds_df = embedding(data, 'description')
    x = preprocess(embeds_df, cfg)
    model_uri = f'models:/{cfg.mlflow.model_name}/{cfg.mlflow.model_ver}'
    model = catboost.load_model(model_uri)
    preds = model.predict(x)
    data['prediction'] = preds
    output_path = str(repo.index.stages[2].outs[0])
    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
