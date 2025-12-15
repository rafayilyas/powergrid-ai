from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

from training.preprocess import preprocess
from training.train_regression import train_regression
from training.train_classification import train_classification
from training.train_timeseries import train_timeseries


@task(retries=2, retry_delay_seconds=10)
def t_preprocess():
    return preprocess()


@task
def t_train_regression():
    return train_regression()


@task
def t_train_classification():
    return train_classification()


@task
def t_train_timeseries():
    return train_timeseries()


@flow(name="train-all")
def train_all():
    t_preprocess()
    r = t_train_regression()
    c = t_train_classification()
    ts = t_train_timeseries()
    return {"regression": r, "classification": c, "timeseries": ts}


if __name__ == '__main__':
    print("Running Prefect flow to train models")
    print(train_all())
