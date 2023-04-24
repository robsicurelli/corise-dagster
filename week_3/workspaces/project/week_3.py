from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
    String,
)
from workspaces.config import REDIS, S3, S3_FILE
from workspaces.project.sensors import get_s3_keys
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

@static_partitioned_config(partition_keys=[str(p) for p in range(1, 11)])
def docker_config(partition_key: int):
    return {
        "resources": {
            "s3": {"config": S3},
            "redis": {"config": REDIS},
        },
        "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}},
    } 

machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config={
        "ops": {"get_s3_data": {"config": {"s3_key": S3_FILE}}},
    },
    resource_defs={
        "s3": mock_s3_resource,
        "redis": ResourceDefinition.mock_resource(),
    }
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    op_retry_policy=RetryPolicy(max_retries=10, delay=1),
    config=docker_config,
    resource_defs={
        "s3": s3_resource,
        "redis": redis_resource,
    }
)

machine_learning_schedule_local = ScheduleDefinition(
    job=machine_learning_job_local,
    cron_schedule="*/15 * * * *"
)


@schedule(job=machine_learning_job_docker, cron_schedule="0 * * * *")
def machine_learning_schedule_docker():
    scheduled_datetime = context.scheduled_execution_time.strftime("%Y-%m-%d %H:%M:%S")
    for partition_key in docker_config.get_partition_keys():
        yield RunRequest(
            run_key=partition_key,
            run_config=docker_config.get_run_config(partition_key),
            tags={"scheduled_datetime": scheduled_datetime, "partition": partition_key},
        )


@sensor(job=machine_learning_job_docker)
def machine_learning_sensor_docker():
    new_files = get_s3_keys(
        S3["bucket"],
        "prefix",
        endpoint_url=S3["endpoint_url"],
    )

    if not new_files:
        yield SkipReason("No new s3 files found in bucket.")
        return
    for new_file in new_files:
        yield RunRequest(
            run_key=new_file,
            run_config={            
                "resources": {
                    "s3": {"config": S3},
                    "redis": {"config": REDIS},
                },
                "ops": {"get_s3_data": {"config": {"s3_key": new_file}}},
            },
        )
