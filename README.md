# HY-320 Software Engineering Project

## Project Overview

This project is a web-based application for data extraction and analysis, built using **Streamlit**. The application allows users to upload datasets, visualize data, perform dimensionality reduction (PCA, UMAP), and apply machine learning techniques such as classification. Users can also select important features, classify datasets, and compare results using various performance metrics.

## Key Features

-   **Data Upload**: Supports CSV, Excel, and TSV formats.
-   **Data Visualization**: Includes PCA, UMAP, and various exploratory data analysis (EDA) plots.
-   **Feature Selection**: Uses SelectKBest for selecting the most important features.
-   **Classification**: Implements K-Nearest Neighbors (KNN) and Random Forest classifiers for analysis.
-   **Results Comparison**: Displays performance metrics such as accuracy, F1-Score, and ROC-AUC for comparing the effectiveness of feature selection.

## Technologies Used

-   **Python 3.12**
-   **Streamlit**
-   **Pandas, Scikit-learn, Plotly**
-   **Docker**

## How to Install and Run the Project

### Prerequisites

-   Docker installed on your machine. You can download Docker from [here](https://www.docker.com/products/docker-desktop).

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/Vafakos/HY-320-Software-Engineering-Project.git
```

### Build and Run with Docker

1. **Build the Docker Image**:
   After navigating to the project directory, build the Docker image using the `Dockerfile` provided in the repository:

    ```bash
    docker build -t my-streamlit-app .
    ```

2. **Run the Docker Container**:
   Once the image is built, run the Docker container, exposing the necessary port:

    ```bash
    docker run -p 8501:8501 my-streamlit-app
    ```

3. **Access the Application**:
   Open your browser and navigate to `http://localhost:8501` to access the application.

## Application Structure

-   **Upload Tab**: Allows users to upload datasets for analysis.
-   **Visualization Tab**: Visualize data using PCA, UMAP, and EDA plots.
-   **Feature Selection Tab**: Select important features using SelectKBest.
-   **Classification Tab**: Apply machine learning models (KNN, Random Forest) and compare performance metrics.
-   **Info Tab**: Provides details about the application and development team.

## Team Contributions

-   **Vafakos Charalampos**:

    -   Implemented file upload and data preprocessing functionality.
    -   Developed PCA visualization and integrated classifiers for the original dataset.
    -   Handled feature selection using SelectKBest.
    -   Managed the GitHub repository and Docker setup.

-   **Filippou Ionas**:
    -   Organized data structure and added labels.
    -   Developed UMAP visualization and assisted in feature selection.
    -   Implemented classification for reduced datasets and compared results.
    -   Assisted with Docker setup and report writing.
