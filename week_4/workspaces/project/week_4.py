from datetime import datetime
from typing import List

from dagster import (
    AssetSelection,
    Nothing,
    OpExecutionContext,
    ScheduleDefinition,
    String,
    asset,
    define_asset_job,
    load_assets_from_current_module,
    static_partitioned_config,
    Definitions,
)
from workspaces.types import Aggregation, Stock
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.resources import redis_resource, s3_resource


@asset(
    config_schema={"s3_key": String},
    op_tags={"kind": "s3"},
    required_resource_keys={"s3"},
)
def get_s3_data(context: OpExecutionContext):
    stocks = [
        Stock.from_list(record)
        for record in context.resources.s3.get_data(context.op_config["s3_key"])
    ]

    assert len(stocks) > 0, f"No stocks found in file: {file_name}"

    return stocks


@asset()
def process_data(context: OpExecutionContext, get_s3_data: List[Stock]):
    high_stock = max(get_s3_data, key=lambda s: s.high)
    return Aggregation(date=high_stock.date, high=high_stock.high)


@asset(op_tags={"kind": "redis"}, required_resource_keys={"redis"})
def put_redis_data(context: OpExecutionContext, process_data: Aggregation):
    context.resources.redis.put_data(str(process_data.date), str(process_data.high))


@asset(op_tags={"kind": "s3"}, required_resource_keys={"s3"})
def put_s3_data(context: OpExecutionContext, process_data: Aggregation):
    context.resources.s3.put_data(
        f"run_{int(datetime.utcnow().timestamp())}_aggregation.json", 
        process_data
    )

@static_partitioned_config(partition_keys=[str(p) for p in range(1, 11)])
def docker_config(partition_key: int):
    return {
        "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}},
    } 


machine_learning_asset_job = define_asset_job(
    name="machine_learning_asset_job",
    config=docker_config,
)

project_assets = load_assets_from_current_module()

machine_learning_schedule = ScheduleDefinition(job=machine_learning_asset_job, cron_schedule="*/15 * * * *")

definition = Definitions(
    resources={
        "s3": s3_resource.configured(S3),
        "redis": redis_resource.configured(REDIS),
    },
    assets=[
        *project_assets,
    ],
    schedules=[machine_learning_schedule],

)