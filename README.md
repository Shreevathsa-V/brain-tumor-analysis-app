# Brain Tumor Analysis Platform

This is a full-stack web application designed for the classification and segmentation of brain tumors from MRI scans. The platform uses a deep learning model to analyze uploaded images and provides a prediction of the tumor type (Glioma, Meningioma, Pituitary, or No Tumor) along with a segmentation mask highlighting the potential tumor area.

---

## Features

-   **Accurate Classification**: Utilizes a U-Net based model built with TensorFlow/Keras to classify brain tumors into four categories.
-   **Tumor Segmentation**: Highlights the predicted tumor region on the original MRI scan.
-   **Batch Processing**: Allows users to upload and analyze multiple scans at once.
-   **Confidence Scoring**: Provides a confidence score for each prediction, indicating the model's certainty.
-   **Modern UI**: A clean, responsive, and user-friendly interface built with React.js.
-   **Decoupled Architecture**: A robust Flask backend serves the model, completely separate from the frontend.

---

## Tech Stack

### Frontend
-   **React.js**: For building the user interface.
-   **CSS**: For custom styling and a professional look and feel.

### Backend
-   **Python**: The core language for the server and model.
-   **Flask**: A micro web framework for the backend API.
-   **TensorFlow/Keras**: For building and serving the deep learning model.
-   **Gunicorn**: A production-ready WSGI server.

---

## Project Structure
/brain-tumor-analyzer
|-- /backend
|   |-- /data
|   |   |-- /Training
|   |   |-- /Testing
|   |-- /frontend
|   |   |-- /src
|   |   |   |-- App.js
|   |   |   |-- App.css
|   |   |   |-- index.js
|   |   |-- package.json
|   |   |-- .gitignore
|   |-- app.py
|   |-- train.py
|   |-- requirements.txt
|   |-- brain_tumor_model.h5
|   |-- .gitignore
|
|-- README.md

---

## Setup and Installation

### Prerequisites
-   Python 3.8+
-   Node.js and npm
-   (For Mac with Apple Silicon) Homebrew for installing `git-lfs`.

### 1. Backend Setup

1.  **Navigate to the backend directory**:
    ```bash
    cd backend
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Frontend Setup

1.  **Navigate to the frontend directory**:
    ```bash
    cd ../frontend
    ```
2.  **Install dependencies**:
    ```bash
    npm install
    ```

---

## Model Training

1.  **Download the Dataset**:
    -   Obtain a brain tumor MRI dataset. The recommended dataset is the "Brain Tumor MRI Dataset" from Kaggle, which is pre-organized into `glioma`, `meningioma`, `pituitary`, and `notumor` folders.
    -   Place the `Training` and `Testing` folders inside the `backend/data` directory.

2.  **Run the Training Script**:
    -   Navigate to the `backend` directory and ensure your virtual environment is active.
    -   Run the training script:
        ```bash
        python3 train.py
        ```
    -   This will train the model and save the best version as `brain_tumor_model.h5` in the `backend` folder.

---

## Usage

You must run both the backend and frontend servers simultaneously in two separate terminals.

1.  **Start the Backend Server**:
    -   Navigate to the `backend` directory.
    -   Run the Gunicorn server (or the Flask development server for easier debugging):
        ```bash
        # For production
        gunicorn --workers 1 --bind 0.0.0.0:8080 app:app
        
        # For development
        python3 app.py
        ```

2.  **Start the Frontend Server**:
    -   Navigate to the `frontend` directory.
    -   Run the React development server:
        ```bash
        npm start
        ```
    -   Your browser will automatically open to `http://localhost:3000`.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
