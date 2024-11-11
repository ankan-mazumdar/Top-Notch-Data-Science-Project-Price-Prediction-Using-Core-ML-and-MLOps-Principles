import click
from pipelines.training_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.client import Client

@click.command()
def main():
    """
    Run the ML pipeline and start both MLflow and ZenML UIs for experiment tracking.
    """
    # Run the pipeline
    run = ml_pipeline()
    
    # Get MLflow tracking URI
    mlflow_tracking_uri = get_tracking_uri()
    print("\nMLflow Tracking URI:", mlflow_tracking_uri)
    print("To start MLflow UI, run:")
    print(f"mlflow ui --backend-store-uri '{mlflow_tracking_uri}'")
    
    # Get ZenML dashboard info
    client = Client()
    print("\nZenML Dashboard:")
    print("1. First run: zenml up")
    print("2. Then visit: http://127.0.0.1:8237")
    print("3. Default username is 'default' (no password needed)")
    
    # Print pipeline run information
    try:
        pipeline_run = client.get_pipeline_run(run.id)
        print(f"\nPipeline Run ID: {pipeline_run.id}")
        print(f"Status: {pipeline_run.status}")
        print(f"Start Time: {pipeline_run.start_time}")
        
        # Print steps information safely
        print("\nSteps Status:")
        steps = pipeline_run.steps
        if isinstance(steps, dict):
            for step_name, step_info in steps.items():
                status = step_info.get('status', 'Unknown')
                print(f"- {step_name}: {status}")
        else:
            print("Steps information not available in expected format")
            
    except Exception as e:
        print(f"\nError getting pipeline details: {str(e)}")
        print("You can still view all details in the ZenML dashboard")

if __name__ == "__main__":
    main()
