from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    String,
    graph,
    op,
    job,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    tags={"kind": "s3"},
    required_resource_keys={"s3"},
)
def get_s3_data(context: OpExecutionContext) -> List[Stock]:
    stocks = [
        Stock.from_list(record)
        for record in context.resources.s3.get_data(context.op_config["s3_key"])
    ]

    assert len(stocks) > 0, f"No stocks found in file: {file_name}"

    return stocks


@op()
def process_data(stocks: List[Stock]) -> Aggregation:
    high_stock = max(stocks, key=lambda s: s.high)
    return Aggregation(date=high_stock.date, high=high_stock.high)


@op(tags={"kind": "redis"}, required_resource_keys={"redis"},)
def put_redis_data(context: OpExecutionContext, aggregation: Aggregation) -> Nothing:
    context.resources.redis.put_data(str(aggregation.date), str(aggregation.high))


@op(tags={"kind": "s3"}, required_resource_keys={"s3"})
def put_s3_data(context: OpExecutionContext, aggregation: Aggregation) -> Nothing:
    context.resources.s3.put_data(
        f"run_{int(datetime.utcnow().timestamp())}_aggregation.json", 
        aggregation
    )


@graph
def machine_learning_graph():
    highest_agg = process_data(
        get_s3_data()
    )

    put_redis_data(highest_agg)
    put_s3_data(highest_agg)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
}

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={
        "s3": mock_s3_resource,
        "redis": ResourceDefinition.mock_resource(),
    }
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker,
    resource_defs={
        "s3": s3_resource,
        "redis": redis_resource,
    }
)
