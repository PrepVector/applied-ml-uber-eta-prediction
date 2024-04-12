# Food Delivery Time Prediction using ML

## Overview

This project aims to implement a predictive model for accurately estimately the food delivery time using ML. The repository is structured with several key components for data analysis, modelling and deployment of the predictive model.

##### Food Delivery Time System

Dataset Link: https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset?select=train.csv

**About Dataset**


<font size="2">Food delivery is a courier service in which a restaurant, store, or independent food-delivery company delivers food to a customer. 

An order is typically made either through a restaurant or grocer's website or mobile app, or through a food ordering company. The delivered items can include entrees, sides, drinks, desserts, or grocery items and are typically delivered in boxes or bags. The delivery person will normally drive a car, but in bigger cities where homes and restaurants are closer together, they may use bikes or motorized scooters.

Prompt and accurate delivery time directly impacts customer satisfaction and influences their overall experience.</font>

## Setting Up the Project

### Prerequisites

- Docker installed on your machine.

### Instructions

1. Clone the repository:

    ```bash
    git clone  https://github.com/PrepVector/Applied-ML.git
    ```

2. Build the Docker image:

    ```bash
    docker build -t food_delivery_time .
    ```

3. Run the Docker container:

    ```bash
    docker run -p 8501:8501 food_delivery_time
    ```

4. Access the Streamlit app in your web browser at [http://localhost:8501](http://localhost:8501).

### Additional Commands

- To enter the Docker container shell:

    ```bash
    docker run -it food_delivery_time /bin/bash
    ```

- To stop the running container:

    ```bash
    docker stop $(docker ps -q --filter ancestor=food_delivery_time)
    ```

Adjust the instructions based on your specific project needs.
