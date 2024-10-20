from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from tgn_stock.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, data_config
from tgn_stock.data import DataRetriever

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "chinese_companies_Oct2024.xlsx",
    output_path: Path = PROCESSED_DATA_DIR,
    dataset_version: str = "1.0.0",
    # ----------------------------------------------
):
    data_retriever = DataRetriever(raw_data_path=input_path, config=data_config)
    df = data_retriever.fetch_data()
    data_retriever.save(df, output_path / f"stock_data_{dataset_version}.parquet")


if __name__ == "__main__":
    app()
