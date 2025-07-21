from src.data_engineering import launch as launch_data_engineering
from src.train import launch as launch_training
from src.model_selection import launch as launch_model_selection


def pipeline():
    """
    Main pipeline function to orchestrate the data engineering, model training,
    and model selection processes.
    """
    # Step 1: Data Engineering
    launch_data_engineering()

    # Step 2: Model Training
    launch_training()

    # Step 3: Model Selection
    launch_model_selection()


if __name__ == "__main__":
    pipeline()
