import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


########################
import os
import pickle

from prefect import get_run_logger, task, flow
from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule

from datetime import date, datetime
from dateutil.relativedelta import relativedelta
########################

@task
def get_paths(date1):
    date1 = datetime.strptime(date1, '%Y-%m-%d').date()
    one_month_ago = date1 - relativedelta(months=2)
    year = one_month_ago.strftime("%Y")
    month = one_month_ago.strftime("%m")
    train_path = f"../../dataset/fhv_tripdata_{year}-{month}.parquet"
    two_month_ago = date1 - relativedelta(months=1)
    year = two_month_ago.strftime("%Y")
    month = two_month_ago.strftime("%m")
    val_path = f"../../dataset/fhv_tripdata_{year}-{month}.parquet"
    return train_path, val_path

@task
def read_data(path):
    if not os.path.exists(path):
        print('Downloading file...')
        fname = os.path.basename(path)
        os.system(f'wget https://nyc-tlc.s3.amazonaws.com/trip+data/{fname} -P ../../dataset/')
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
        # print(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
        # print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    # print(f"The shape of X_train is {X_train.shape}")
    # print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    # print(f"The MSE of training is: {mse}")
    logger.info(f"The MSE of training is: {mse}")

    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    # print(f"The MSE of validation is: {mse}")
    logger.info(f"The MSE of validation is: {mse}")
    return

# @flow
# def main(train_path: str = '../../dataset/fhv_tripdata_2021-01.parquet', 
#            val_path: str = '../../dataset/fhv_tripdata_2021-02.parquet'):



@flow(task_runner=SequentialTaskRunner())
def main(date1=None):
    if date1 is None:
        date1 = date.today()
    train_path, val_path = get_paths(date1).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open(f"models/model-{date1}.bin", "wb") as f_out:
            pickle.dump(lr, f_out)
    with open(f"models/dv-{date1}.bin", "wb") as f_out:
            pickle.dump(dv, f_out)

# main()
# main(date1="2021-08-15")

DeploymentSpec(
    name="cron-schedule-deployment",
    flow=main,
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Asia/Karachi"),
)