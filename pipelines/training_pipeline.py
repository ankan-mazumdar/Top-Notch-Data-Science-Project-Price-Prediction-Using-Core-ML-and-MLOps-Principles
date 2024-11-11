import os
import glob
import shutil
from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline, step


@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="prices_predictor"
    ),
    enable_cache=True
)
def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    try:
        # Find zip files in the data directory
        zip_files = glob.glob("data/*.zip")
        
        if not zip_files:
            # If no zip file found, check if CSV exists in extracted_data
            if os.path.exists("extracted_data/AmesHousing.csv"):
                print("No zip file found, but using existing CSV in extracted_data/")
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                # Create zip file from existing CSV
                shutil.make_archive("data/AmesHousing", 'zip', "extracted_data", "AmesHousing.csv")
                file_path = "data/AmesHousing.zip"
            else:
                raise FileNotFoundError("Neither zip file nor CSV file found. Please ensure data is available.")
        else:
            file_path = zip_files[0]
            print(f"Using existing zip file: {file_path}")

        # Data Ingestion Step
        raw_data = data_ingestion_step(file_path=file_path)

        # Handling Missing Values Step
        filled_data = handle_missing_values_step(raw_data)

        # Feature Engineering Step
        engineered_data = feature_engineering_step(
            filled_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
        )

        # Outlier Detection Step
        clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

        # Data Splitting Step
        X_train, X_test, y_train, y_test = data_splitter_step(clean_data, target_column="SalePrice")

        # Model Building Step
        model = model_building_step(X_train=X_train, y_train=y_train)

        # Model Evaluation Step
        evaluation_metrics, mse = model_evaluator_step(
            trained_model=model, X_test=X_test, y_test=y_test
        )

        return model

    except FileNotFoundError as e:
        print(f"Data Error: {str(e)}")
        print("Please ensure either AmesHousing.zip exists in data/ directory or AmesHousing.csv exists in extracted_data/ directory")
        raise
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        print("Please check logs for more details")
        raise

    finally:
        print("Pipeline execution completed")


if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
