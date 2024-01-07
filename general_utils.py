import json
import boto3
import mgzip
import random
import itertools
import dill as pkl
import multiprocessing
import more_itertools as mit


def read_json(path):
    with open(path) as infile:
        return json.load(infile)


def unload_pickle(obj, path, compress=False):
    open_func = mgzip.open if compress else open
    with open_func(path, "wb") as outfile:
        pkl.dump(obj, outfile)


def load_pickle(path, compress=False):
    open_func = mgzip.open if compress else open
    with open_func(path, "rb") as infile:
        return pkl.load(infile)


def _read_from_S3(bucket, folder, file_name, loads_func):
    s3_client = boto3.client("s3")
    return loads_func(
        s3_client.get_object(Bucket=bucket, Key=f"{folder}/{file_name}")["Body"].read()
    )


def unload_pickle_from_S3(bucket, folder, file_name):
    return _read_from_S3(bucket, folder, file_name, pkl.loads)


def read_json_from_S3(bucket, folder, file_name):
    return _read_from_S3(bucket, folder, file_name, json.loads)


def task_to_parallelize(fun, inputs):
    return [fun(*x) for x in inputs]


def parallelize_task(fun, inputs, njobs=3, shuffle=True):
    if shuffle:
        random.shuffle(inputs)
    chunk_inputs = [(fun, list(c)) for c in mit.divide(njobs, inputs)]
    with multiprocessing.Pool(njobs) as pool:
        out = pool.starmap(task_to_parallelize, chunk_inputs)
    return list(itertools.chain(*out))
