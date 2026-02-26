#!/usr/bin/env bash
SPARK_SUBMIT=${SPARK_SUBMIT:-spark-submit}
CONF=${1:-configs/spark-cpu.conf}
APP=${2:-examples/spark_etl_job.py}
$SPARK_SUBMIT --properties-file $CONF $APP
