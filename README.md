# Top-Notch-Data-Science-Project-Price-Prediction-Using-Core-ML-and-MLOps-Principles

  ![image](https://github.com/user-attachments/assets/a28156c0-7703-40a7-b0b6-a4acde54b54d)

  ![image](https://github.com/user-attachments/assets/c301c0bf-d337-42aa-a16e-aa36b836a504)

  <img width="1440" alt="Screenshot 2024-11-11 at 12 22 00 AM" src="https://github.com/user-attachments/assets/5cebecba-d688-4a27-9c85-428f9198f62d">

  <img width="712" alt="Screenshot 2024-11-11 at 12 21 52 AM" src="https://github.com/user-attachments/assets/f1e03242-de7c-4a93-9ee2-44c1cf3e825f">




<img width="1396" alt="Screenshot 2024-11-11 at 12 21 40 AM" src="https://github.com/user-attachments/assets/16af9842-1b45-43ba-b623-d4f4a8e65d9b">




  <img width="1413" alt="Screenshot 2024-11-11 at 12 20 55 AM" src="https://github.com/user-attachments/assets/d9fd3692-9a31-43f4-8908-b1b6a04eb886">

  <img width="812" alt="Screenshot 2024-11-11 at 12 38 22 AM" src="https://github.com/user-attachments/assets/8e366fd8-297e-41ae-932e-8edb4f39ca53">

<img width="1406" alt="Screenshot 2024-11-11 at 12 43 17 AM" src="https://github.com/user-attachments/assets/197b46e0-74e1-43cb-8e8e-52c6bcd16f61">

<img width="1436" alt="Screenshot 2024-11-11 at 12 42 31 AM" src="https://github.com/user-attachments/assets/a8075d49-0e7b-462e-9d08-ba2407266cea">

<img width="1432" alt="Screenshot 2024-11-11 at 12 43 37 AM" src="https://github.com/user-attachments/assets/2f2be932-aadd-415d-8233-fd978cfa5d46">
  
  <img width="633" alt="Screenshot 2024-11-11 at 12 45 45 AM" src="https://github.com/user-attachments/assets/8c497968-2b57-4a98-bc5c-eec334d0ff59">




## Summary
**"Building a Top-Notch Data Science Project: House Price Prediction Using Core ML and MLOps Principles"**

This project demonstrates the combined use of core ML techniques and MLOps principles to build a robust, production-grade pipeline for house price prediction. It includes CI/CD, experiment tracking, and design patterns that ensure the model is reproducible, scalable, and aligned with industry standards.

## Key Components

### Core ML
- **Focus on Data Understanding**: Time is allocated for thorough data analysis and assumption testing, which significantly influences algorithm selection and feature engineering steps.
- **Model Compatibility**: Data validation and assumption checks are integrated to ensure the chosen model performs effectively on the dataset.

### MLOps
- **Comprehensive MLOps Framework**: Integrates tools
-  ZenML (easy to use orchestrated MLOps framework seamlessly integrating other ML tools/workflows to build fullstack of it, relatively easy to understand, we understand the pipelins workflows) and MLFlow (inegration of ZenML where we use to track our deployments for a production-ready ML pipeline.

  ![image](https://github.com/user-attachments/assets/fd9e9c9c-49d3-47c9-ae42-31b2a8109b71)

- **Pipeline Automation**: CI/CD pipelines automate the flow from data ingestion through deployment, enhancing reproducibility and robustness.

### Code Quality
- **Adoption of Design Patterns**: Uses design patterns (Factory, Strategy, Template) across modules, making code scalable, modular, and easier to maintain.
<img width="237" alt="Screenshot 2024-10-28 at 4 23 43 AM" src="https://github.com/user-attachments/assets/e034e1e2-147f-4e75-9906-886212345b35">

Factory Design Pattern

Imagine you run a coffee shop. Customers can order different types of coffee, but the process of
making coffee follows a similar pattern. You have a general coffee-making machine (the factory)
that can be used to make different types of coffee (products) like Espresso, Latte, or Cappuccino.
CoffeeMachine (Factory): Has a method to make coffee
• Espresso, Latte, Cappuccino (ConcreteProducts): Different types of coffee that can be
made by the machine.

Strategy Pattern

Imagine you're developing an e-commerce application. Customers can choose different payment
methods like Credit Card, PayPal, or Bitcoin. Each payment method has a different implementation,
but the overall process is the same: the customer pays for the order.
• PaymentMethod (Strategy): An interface that defines how payments are processed.
• CreditCardPayment, PayPalPayment, BitcoinPayment (ConcreteStrategies): Different
implementations of payment processing.
• ShoppingCart (Context): Uses a payment method to process a customer's payment.

Template Pattern

Real-World Analogy:
Imagine you run a restaurant with a set menu for different cuisines. Each cuisine (like Italian,
Chinese, or Indian) has a specific sequence of courses: appetizer, main course, dessert, and
beverage. The sequence of serving these courses is the same, but the dishes served at each step
vary depending on the cuisine.
For example, in an Italian meal, the appetizer might be bruschetta, the main course could be pasta
dessert might be tiramisu, and the beverage could be a glass of wine. In a Chinese meal, the
appetizer could be spring rolls, the main course might be stir-fried noodles, dessert could be
fortune cookies, and the beverage could be tea.
The template here is the overall dining sequence: appetizer, main course, dessert, and beverage.
The customizable steps are the specific dishes served at each stage, which change based on the
cuisine.

---
## Why Understanding and validating data are crucial in data science and machine learning
Note, we will implement a single model for this project, however we should be testing it with different models to test in future.
- Many people focus only on training and evaluating the model, without understanding the underlying data
- It's important to maintain a robust model over time and to continually improve it with deeper insights from the data
- Plan includes project structure, data sources, model deployment, and design patterns like strategy, template, and factory.
- Utilize design patterns like Factory design pattern for code reproducibility and scalability
- Understanding the factory pattern in AI projects
- The factory pattern involves creating a class with a preparation stage for different products like coffee types
- It helps in creating different product instances and calling preparation methods for each product
- Integrating CI/CD pipelines and using MLflow for experiment tracking and ZenML for orchestration
- Implementing the ingest method for data processing
- The ingest method takes the file path as input and returns a data frame.
- Adding type checks and docstrings for clarity and readability of the code.
- Handling non-zip files and extracting data from a zip file
- Professionals verify file type before extraction for error handling
- Using the zip file library to extract data and ensure presence of CSV files
- Using abstract classes for data ingestion
- Abstract classes can be extended to handle specific data formats like JSON.
- Automating the process of determining file extensions for data ingestion.
- Understanding Strategy Pattern in E-commerce Application Development
- Strategy pattern allows for different payment methods in e-commerce apps while keeping the payment process consistent.
- By creating a strategy interface and multiple implementations, developers can handle various payment gateways efficiently.
- Creating scalable data analysis using Analyze Source and EDA
- Implementing various analysis techniques like data inspection, missing value analysis, and multivariate analysis
- Utilizing Analyze Source for logic implementation and EDA for data visualization

- Creating an abstract data inspection strategy
- The basics of data inspection involve inspecting the data frame
- The abstract base class defines a common interface for data inspection subclasses

- Executing data inspection strategy
- Setting the strategy and executing it by calling the 'do_inspect' method on the data frame.
- Utilizing summary statistics inspection strategy without reinstantiating the data inspector.

- Exploring and understanding dataset insights
- Ability to view and explore numerical and categorical columns with detailed summary statistics
- Insights on numerical features such as target variable sale price with mean and standard deviation

- Data inspection reveals outliers, missing values, and skewness
- Need to analyze distribution of sale price and other numeric features
- Important to handle outliers and missing values for better algorithm performance

Teaching about template design patterns and missing value analysis.
- Template design pattern is used frequently in sessions and is compared to different cuisines having their own sequence of courses.
- The missing value analysis will be covered in the upcoming sessions.

Creating a template for missing value analysis
- The abstract class defines a plan for identifying and visualizing missing values
- Concrete class implements methods to identify and visualize missing values in a data frame

 Creating a heat map using Seaborn library in Python.
- Utilize Seaborn to create a heat map for analyzing correlations and value analysis.
- Data frame with Boolean values to indicate missing values when creating the heat map.

Understanding missing data distribution
- There are two types of missing data distribution - randomly distributed and structured missingness.
- Structured missingness may indicate a nonrandom pattern, which could suggest a data collection issue or non-applicability of features.

Data preprocessing steps for regression analysis
- Outliers handling and categorical encoding are crucial
- Feature engineering involving combining related features and transforming skewed data is necessary

Visualizing Target variable with kernel density estimate and histogram bins
- Kernel density estimate creates a smooth version of histogram to better estimate data distribution.
- The number of bins in the histogram affects the granularity and level of detail in the visualization.
- 
- Exploring how to analyze and interpret relationships between numerical features
- Implementing numerical versus numerical analysis to visualize relationships

- The module enforces any subclass to implement the handle method for handling missing values
- Two strategies are discussed - dropping missing values and filling missing values using various methods such as mean, median, mode, or constant values

 Handling missing values in data processing using various strategies.
- Mean, median, mode, or constant values used to fill missing numeric data depending on the method chosen.
- Context class created to delegate the task of handling missing values with the flexibility to switch between strategies.

 Running a machine learning pipeline with various steps
- The process involves handling warnings and initiating a new run for the ML pipeline.
- The pipeline includes steps such as handling missing values, feature engineering, outlier detection, and model training and evaluation.

Feature engineering and outlier detection are important for data normalization.
- Feature engineering involves defining strategies for applying transformations to skewed features in the data.
- Outlier detection can help in reducing skewness in the data by applying appropriate transformations.

 Explaining feature engineering and strategy application
- Using one hot encoding in feature engineering
- Applying lock transformation, scaling, and one hot encoding in the training Pipeline

Implement zcore outlier detection method
- Zcore outlier detection is a statistical method using standard deviation to identify outliers in a data set.
- Zcore of zero indicates data point is at the mean, zcore of one indicates one standard deviation above the mean.

Explanation of handling outliers using strategies like removing and capping
- Capping changes outlier values to nearest non-outlier data point
- Set upper and lower bounds to handle outliers effectively

Importance of Model Building in AI Project
- Data splitting is essential to check the model performance.
- Feature engineering involves strategies like logging and standard scaling.

Importance of Model Building in AI Project
- Data splitting is essential to check the model performance.
- Feature engineering involves strategies like logging and standard scaling.

 Model building involves importing libraries and setting up configurations.
- Linear regression pipeline is imported from scikit-learn, including the regressor mixin.
- Abstract and concrete classes are created for model building strategies, utilizing pandas data frames for training.
 Implementing standard scaler for feature scaling
- Standard scaler ensures zero mean and unit variance for features.
- Using standard scaling in the pipeline for optimal model training and prediction.

Setting up the ML flow stack for the project.
- Default orchestrator, experiment tracker, ml4 tracker, model deploy, ml flow prices, and artifact store should be set.
- Imported ml flow, pandas, and regressor mixing for indicating the output.

Explanation of pipeline creation in AI project
- Pipeline named 'price predictor' used to transform and scale data
- Model artifact created for training and prediction using categorical and numerical columns

Understanding input required by the model after processing
- Logging the column that the model expects is helpful for understanding the input required by the model after processing, especially when using one hot encoding.
- Tracking and comparing different experiments with feature transformations is essential and can be achieved by extracting every run.
 
## Detailed Project Structure

### Data Loading & Ingestion
- **Why Factory Design Pattern**: The Factory pattern is utilized to handle various data formats dynamically. It provides flexibility in data ingestion by creating different "products" (i.e., data types like CSV, JSON, zip) from a single "factory" (data ingestion class). This design ensures scalability and the capacity to adapt to additional data types with minimal changes.
- **How It’s Better**: Unlike hardcoded file handlers, this approach uses a generalized interface to recognize and process different data formats automatically. It saves time and simplifies adding new formats by extending a base class instead of editing existing code.
- **Implementation**:
  - **Data Source Versatility**: The Factory pattern manages all data sources and formats through its extensible interface.
  - **Type Checking & Error Handling**: Validates file types before ingestion, preventing issues that arise from unsupported file types. For instance, zip files are verified and extracted using Python’s `zipfile` library, ensuring only valid data is processed.
- **Other Options & Comparison**: Traditional hardcoded ingestion scripts can handle specific formats but lack scalability, require significant manual code adjustments, and are challenging to maintain across multiple formats.

### Exploratory Data Analysis (EDA)
- **Why Julius AI**: Julius AI is incorporated for fast, thorough data analysis, quick insights, and visualizations, significantly reducing the time spent on manual EDA. It supports interactive data visualizations and quick experimentation, allowing for a more insightful and streamlined approach to data exploration.
- **How It’s Better**: Compared to traditional manual EDA, Julius AI saves considerable time and effort by automating many exploratory steps. It includes community suggestions and pre-built workflows that aid in feature selection, multivariate analysis, and identifying relationships between variables, thus creating a solid foundation for model-building decisions.

- ![image](https://github.com/user-attachments/assets/c8daa82f-75e6-40a7-a7b1-268895b99d09)

- **EDA Strategy Pattern**: The Strategy pattern allows easy switching between various data inspection strategies, such as summary statistics, missing value analysis, and multivariate relationships. By defining these strategies as separate modules, EDA becomes modular and easily extendable.
  - **Scalability**: This structure ensures EDA steps are reusable, well-organized, and easily adaptable to new datasets or analysis needs.
- **Other Options & Comparison**: Performing EDA manually in a Jupyter notebook can work for smaller projects but is labor-intensive and error-prone for larger datasets. Julius AI, paired with the Strategy pattern, offers a faster, structured, and replicable alternative.

### Handling Missing Values & Outliers
- **Why Template Design Pattern**: To manage missing values effectively, a Template pattern is used to create reusable structures for missing value analysis. This approach standardizes handling processes, making it easier to spot missing patterns and structure preprocessing steps accordingly.
  - **Seaborn Heatmaps**: Heatmaps are used to visualize missing data distributions, with structured and random missingness guiding preprocessing decisions.
- **Outlier Detection & Normalization**: Outliers are detected using the Z-score method, which is a statistical approach that flags data points significantly deviating from the mean. This method is reliable for identifying unusual values in continuous data and supports decision-making in feature engineering.

  ![image](https://github.com/user-attachments/assets/bb42d223-7021-405f-9d56-70552db3067c)

- **How It’s Better**:
  - **Template Pattern Benefits**: By setting a base class for missing value analysis, handling missing data becomes modular, adaptable, and standardized, which is particularly useful for larger, complex datasets.
  - **Automated Z-Score Analysis**: Compared to manual outlier checks, Z-score automates and simplifies the identification of outliers, ensuring that skewed values are treated consistently.
- **Other Options & Comparison**: Traditional approaches, such as manual thresholding for outliers, lack standardization and scalability. The Template pattern with Z-score outlier detection provides a robust and systematic alternative, especially in production settings where consistency is critical.

### Feature Engineering
- **Why Use One-Hot Encoding & Standard Scaling**: One-hot encoding handles categorical variables efficiently, transforming them into a format that machine learning models can interpret. Standard scaling ensures that numerical features have a mean of zero and a standard deviation of one, making them suitable for algorithms sensitive to feature scaling.
  - **Log Transformations**: Log transformations are applied to certain features to reduce skewness and improve model interpretability.
- **How It’s Better**:
  - **Standard Scaling**: Normalizes the data, which is crucial for models that rely on distance-based calculations (e.g., linear regression, SVM).
  - **Automated Feature Pipeline**: The transformation process is embedded into the pipeline, ensuring the same transformations are applied every time, regardless of new data.
- **Other Options & Comparison**: While one-hot encoding and standard scaling are common, their automation within a pipeline ensures consistency across multiple runs, reducing the risk of human error and enhancing model reproducibility.

### Modeling & Deployment
- **Experiment Tracking & Deployment**: Using MLFlow, experiments are tracked meticulously, providing insights into model performance across different runs and configurations. This allows for quick model selection based on tracked metrics.
- **CI/CD Pipeline**: Automated testing and deployment ensure that any new model iteration is rigorously tested and ready for production without manual intervention.
- **How It’s Better**: Traditional deployment methods lack robust tracking and version control, making it challenging to reproduce or improve models. The automated CI/CD pipeline with experiment tracking addresses these limitations, creating a streamlined deployment process.
- **Other Options & Comparison**: Manual deployment lacks the scalability and consistency required for production-grade models. MLFlow and ZenML create an effective ecosystem for deploying and maintaining models, even as new data or changes are introduced.

---

## Tools & Libraries

- **ZenML**: Orchestrates MLOps pipelines, improving the management and scalability of machine learning workflows.

  ![image](https://github.com/user-attachments/assets/a120c16e-4733-4154-8d5a-62063a8f9efd)

  ![image](https://github.com/user-attachments/assets/fa7ce889-b0f8-48ca-afb1-54bfb447ea29)


- **MLFlow**: Tracks experiments and manages model deployment, essential for production-level tracking and deployment consistency.

  ![image](https://github.com/user-attachments/assets/5c3b1cbc-2148-4829-be64-074941bfcbec)

- **Julius AI**: Assists in EDA, providing quick insights, visualizations, and supporting in-depth data analysis.
  
- ![image](https://github.com/user-attachments/assets/74780c25-1bff-4819-a781-b6373eed5c9a)

- **Pandas & NumPy**: Core libraries for data manipulation.
- **Seaborn**: Used for visualizations during EDA, especially for correlation analysis and missing value heatmaps.

---

## How to Run the Project
### I will add further screenshots how the output looks post execution

1. **Clone the Repository**:
   ```bash
   git clone <repo-link>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **ZenML and MLflow Setup Guide

```markdown


This guide provides step-by-step instructions for setting up ZenML with MLflow integration.

## Table of Contents
- [Installation](#installation)
- [Basic Setup](#basic-setup)
- [Component Registration](#component-registration)
- [Stack Configuration](#stack-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [Advanced Configuration](#advanced-configuration)
- [Clean Up](#clean-up)

## Installation

```bash
# Install required packages
pip install zenml
pip install mlflow>=2.1.1
```

## Basic Setup

Initialize ZenML repository:
```bash
zenml init
```

Check current environment:
```bash
# List all components to see what's already registered
zenml artifact-store list
zenml orchestrator list
zenml experiment-tracker list
zenml stack list
```

## Component Registration

Register required components:
```bash
# Register local artifact store
zenml artifact-store register local_store --flavor=local

# Register local orchestrator
zenml orchestrator register local_orchestrator --flavor=local

# Register MLflow experiment tracker
zenml experiment-tracker register mlflow --flavor=mlflow
```

## Stack Configuration

Create and set up MLflow stack:
```bash
# Register the stack with components
zenml stack register mlflow_stack \
    -a local_store \
    -o local_orchestrator \
    -e mlflow

# Set as active stack
zenml stack set mlflow_stack
```

## Verification

Verify the setup:
```bash
# Check registered stacks
zenml stack list

# Start MLflow UI (optional)
zenml up
```

## Troubleshooting

### Component Already Exists
```bash
# Delete existing components if needed
zenml artifact-store delete local_store
zenml orchestrator delete local_orchestrator
zenml experiment-tracker delete mlflow
```

### Check Component Details
```bash
zenml artifact-store describe local_store
zenml orchestrator describe local_orchestrator
zenml experiment-tracker describe mlflow
```

### Stack Issues
```bash
# Check stack configuration
zenml stack describe mlflow_stack

# Delete and recreate if needed
zenml stack delete mlflow_stack
```

## Best Practices

1. **Before Starting:**
   - Clean up any existing configurations if starting fresh
   - Check all required packages are installed
   - Verify you're in the correct directory

2. **During Setup:**
   - Register components one at a time
   - Verify each component after registration
   - Use descriptive names for components and stacks

3. **After Setup:**
   - Verify stack activation
   - Test MLflow tracking
   - Check MLflow UI accessibility

## Advanced Configuration

### Custom MLflow Configuration
```bash
# Configure MLflow with specific tracking URI
zenml experiment-tracker register mlflow \
    --flavor=mlflow \
    --tracking_uri=sqlite:///mlflow.db
```

### Custom Artifact Store Path
```bash
# Configure local artifact store with specific path
zenml artifact-store register local_store \
    --flavor=local \
    --path=/custom/path/to/artifacts
```

## Clean Up

Remove all configurations:
```bash
# Delete stack and components
zenml stack delete mlflow_stack
zenml artifact-store delete local_store
zenml orchestrator delete local_orchestrator
zenml experiment-tracker delete mlflow

# Clean MLflow artifacts
rm -rf mlruns/
rm mlflow.db
```

## Common Issues and Solutions

1. **Component Registration Fails:**
   - Ensure you have the correct permissions
   - Check if component already exists
   - Verify the flavor is supported

2. **Stack Registration Fails:**
   - Ensure all components are registered first
   - Check component names are correct
   - Verify all required components are specified

3. **MLflow Integration Issues:**
   - Ensure MLflow is properly installed
   - Check if MLflow server is running
   - Verify artifact store is properly configured

4. **Run Data Ingestion**:
   ```bash
   python ingest_data.py
   ```
5. **Run EDA**:
   ```bash
   jupyter notebook eda.ipynb
   ```
6. **Train and Evaluate the Model**:
   ```bash
   python train_model.py
   ```
7. **Track Experiments and Deploy**:
   MLFlow is used for managing experiments and deploying the trained model into production.




