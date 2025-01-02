import os
from pathlib import Path

import typer
from typing_extensions import Annotated
from loguru import logger

import pandas as pd
import time

from tgn_stock.config import PROCESSED_DATA_DIR, graph_build_config
from tgn_stock.graph import GraphManager

app = typer.Typer()


@app.command()
def main(
    dataset_path: Annotated[
        Path, typer.Option(help="The dataset path")
    ] = PROCESSED_DATA_DIR / "stock_data_2.0.0.parquet",
    freq: Annotated[
        str,
        typer.Option(help="The daily frequency (e.g. 30D) for graph snapshot updating"),
    ] = "30D",
    output_path: Annotated[
        Path, typer.Option(help="The output path where graph snapshots are saved")
    ] = PROCESSED_DATA_DIR / "influence_graphs",
    graph_version: Annotated[
        str, typer.Option(help="The influence graph")
    ] = "1.0.0",
):
    """
    Builds stock relationship graphs over a specified time frequency using the stock data provided in a DataFrame.
    """
    # Paths
    assert dataset_path.exists(), f"Path {dataset_path} does not exist"
    assert output_path.exists(), f"Path {output_path} does not exist"
    output_path = output_path / graph_version
    output_path.mkdir(exist_ok=True, parents=True)
    # Load Data
    df = pd.read_parquet(dataset_path)
    assert not df.isna().sum().any(), "There are some NaN in provided dataset"

    # Unpack config
    features = graph_build_config["features"]
    features_to_norm = graph_build_config["features_to_norm"]
    cls_name = graph_build_config["cls_name"]
    cls_params = graph_build_config["cls_params"]
    lambda_weight = graph_build_config["lambda_weight"]
    min_data_points = graph_build_config["min_data_points"]
    train_size = graph_build_config["train_size"]

    # Initialize graph manager
    grap_manager = GraphManager(
        stock_data=df,
        output_path=output_path,
    )

    # Build graph's snapshot every N days
    ref_date = df.index.max()
    start_date = df.index.min()

    # Generate date ranges at freq-day intervals (counting backward from the latest date)
    date_ranges = pd.date_range(end=ref_date, start=start_date, freq=freq)

    # Initialize counter and start time for monitoring progress
    iteration_count = 0
    total_intervals = len(date_ranges)
    start_time = time.time()

    # Create a dataframe with stock list available at each reference date
    df_stock_list = pd.DataFrame(
        df.reset_index().groupby("Date")["Ticker"].agg(list)
    ).reset_index()
    df_stock_list["tot"] = df_stock_list["Ticker"].apply(len)

    # Log info
    logger.info(
        f"Build graph snapshots with {freq} freq for {total_intervals} timestamps starting from {date_ranges[-1].strftime('%Y-%m-%d')}..."
    )
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(
        f"Args:\n{features=}\n{features_to_norm=}\n{cls_name=}\n{cls_params=}\n{lambda_weight=}\n{min_data_points=}\n{train_size=}"
    )

    # Iterate over the generated N-day date ranges in reverse order (latest to oldest)
    for i in range(len(date_ranges) - 1, 0, -1):
        iteration_count += 1
        iteration_start_time = time.time()

        # Current reference date and start date for this 30-day window
        ref_date = date_ranges[i]
        start_window_date = date_ranges[i - 1]

        # Filter data for the 30-day window by date range
        # This will get all rows that correspond to dates between start_window_date and ref_date
        window_data = df[(df.index >= start_window_date) & (df.index <= ref_date)]

        if window_data.empty:
            continue  # Skip if no data is available for this window

        # Now, ensure that the reference date corresponds to the most recent available data in this window
        # For example, use the latest available trading day in the window
        ref_date = window_data.index.max()
        stock_list = df_stock_list[df_stock_list["Date"] == ref_date]["Ticker"].item()

        # Build the graph for the current N-day interval
        check_date = ref_date.strftime("%Y-%m-%d")
        filename = f"stock_graph_{check_date}.pickle"
        if filename not in os.listdir(output_path):
            grap_manager.build_graph_for_reference_date(
                reference_date=ref_date,
                stock_list=stock_list,
                features=features,
                features_to_norm=features_to_norm,
                cls_name=cls_name,
                cls_params=cls_params,
                lambda_weight=lambda_weight,
                min_data_points=min_data_points,
                train_size=train_size,
            )
        else:
            # If ref_date is in the output path, we assume the influence graph is already computed for that day
            logger.info(f"Influence graph already computed for {filename}. Skipping {check_date} ...")
            continue

        # Calculate iteration time and total elapsed time
        iteration_time = time.time() - iteration_start_time
        total_elapsed_time = time.time() - start_time

        # Estimate remaining time based on the average time per iteration
        avg_iteration_time = total_elapsed_time / iteration_count
        remaining_iterations = total_intervals - iteration_count
        estimated_remaining_time = avg_iteration_time * remaining_iterations

        # Print progress and time estimate
        logger.info(
            f"Iteration {iteration_count}/{total_intervals} completed. Time for this iteration: {iteration_time:.2f} seconds"
        )
        logger.info(
            f"Estimated remaining time: {estimated_remaining_time / 60:.2f} minutes"
        )

    logger.success("Graph snapshots computation completed.")


if __name__ == "__main__":
    app()
