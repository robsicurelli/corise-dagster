import csv
from datetime import datetime
from typing import Iterator, List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    String,
    job,
    op,
    usable_as_dagster_type,
)
from pydantic import BaseModel


@usable_as_dagster_type(description="Stock data")
class Stock(BaseModel):
    date: datetime
    close: float
    volume: int
    open: float
    high: float
    low: float

    @classmethod
    def from_list(cls, input_list: List[str]):
        """Do not worry about this class method for now"""
        return cls(
            date=datetime.strptime(input_list[0], "%Y/%m/%d"),
            close=float(input_list[1]),
            volume=int(float(input_list[2])),
            open=float(input_list[3]),
            high=float(input_list[4]),
            low=float(input_list[5]),
        )


@usable_as_dagster_type(description="Aggregation of stock data")
class Aggregation(BaseModel):
    date: datetime
    high: float


def csv_helper(file_name: str) -> Iterator[Stock]:
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield Stock.from_list(row)


@op(
    config_schema={"s3_key": String},
    tags={"kind": "s3"}
)
def get_s3_data_op(context: OpExecutionContext) -> List[Stock]:
    file_name = context.op_config["s3_key"]

    stocks = list(csv_helper(file_name))
    assert len(stocks) > 0, f"No stocks found in file: {file_name}"

    return stocks


@op()
def process_data_op(stocks: List[Stock]) -> Aggregation:
    high_stock = max(stocks, key=lambda s: s.high)
    return Aggregation(date=high_stock.date, high=high_stock.high)


@op(tags={"kind": "redis"})
def put_redis_data_op(aggregation: Aggregation) -> Nothing:
    return Nothing()


@op(tags={"kind": "s3"})
def put_s3_data_op(aggregation: Aggregation) -> Nothing:
    return Nothing()


@job
def machine_learning_job():
    highest_agg = process_data_op(
        get_s3_data_op()
    )

    put_redis_data_op(highest_agg)
    put_s3_data_op(highest_agg)