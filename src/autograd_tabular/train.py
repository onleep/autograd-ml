import hydra
import mlflow
import pandas as pd
from catboost import CatBoostRegressor, Pool
from dvc.repo import Repo
from mlflow import catboost, models
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


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


def preprocess(data: pd.DataFrame, cfg: DictConfig) -> tuple[Pool, Pool, pd.DataFrame]:
    y = data[cfg.data.target_col]
    x = data.drop(columns=cfg.data.drop_cols)
    int_cols = [c for c in x.columns if x[c].dtype == 'int64']
    x[int_cols] = x[int_cols].astype(float)
    cat_cols = [c for c in x.columns if x[c].dtype == 'object']
    x[cat_cols] = x[cat_cols].astype(str)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=cfg.preprocess.test_size,
        random_state=cfg.preprocess.random_state,
    )
    train_pool = Pool(x_train, label=y_train, cat_features=cat_cols)
    val_pool = Pool(x_val, label=y_val, cat_features=cat_cols)
    return train_pool, val_pool, x_val


@hydra.main(config_path='../../configs', config_name='config', version_base=None)
def main(cfg: DictConfig) -> None:
    repo = Repo()
    repo.pull(targets=['download'])
    dataset_path = str(repo.index.stages[0].outs[0])
    data = embedding(pd.read_csv(dataset_path), 'description')
    train_pool, val_pool, x_val = preprocess(data, cfg)
    model = CatBoostRegressor(**cfg.model)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if not exp:
        mlflow.create_experiment(cfg.mlflow.experiment_name)
    elif exp.lifecycle_stage == 'deleted':
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(model.get_params())
        model.fit(train_pool, eval_set=val_pool, verbose=True)
        metrics = model.eval_metrics(val_pool, metrics=list(cfg.train.metrics))
        best_iter = model.get_best_iteration()
        logged = {m: metrics[m][best_iter] for m in cfg.train.metrics}
        logged['BEST_ITER'] = best_iter
        mlflow.log_metrics(logged)
        sign = models.infer_signature(x_val, model.predict(x_val))
        catboost.log_model(
            cb_model=model,
            signature=sign,
            input_example=x_val[:5],
            name=cfg.mlflow.model_name,
            registered_model_name=cfg.mlflow.model_name,
        )


if __name__ == '__main__':
    main()
