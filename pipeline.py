from src.data_engineering import launch as launch_data_engineering
from src.train import launch as launch_training
from src.model_selection import launch as launch_model_selection
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
)


def pipeline():
    """
    Main pipeline function to orchestrate the data engineering, model training,
    and model selection processes.
    """
    # Step 1: Data Engineering
    logger.info("Data engineering started.")
    launch_data_engineering()

    # Step 2: Model Training
    logger.info("Model training started.")
    launch_training()

    # Step 3: Model Selection
    logger.info("Model selection started.")
    launch_model_selection()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    pipeline()
