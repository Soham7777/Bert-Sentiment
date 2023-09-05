
# Sentiment Analysis Project

Welcome to the Sentiment Analysis Project! This project includes sentiment classification of Amazon review data using a combination of deep learning with BERT encoders and a traditional machine learning approach with SVM. Here, you'll find details on both the deep learning and SVM-based deployment, along with example screenshots.

## Deep Learning Approach (BERT with TensorFlow)

### Overview
- The deep learning model in this project is built using TensorFlow and BERT encoders.
- It achieved good predictive accuracy for sentiment classification.

### Jupyter Notebook
- For a detailed walkthrough and code, you can refer to the [Jupyter Notebook](link-to-your-jupyter-notebook.ipynb) provided in this repository.

### SVM Approach
- The project also demonstrates the deployment of a sentiment classification model using Support Vector Machines (SVM) with a linear kernel.
- Deployment with tensor flow model was consuming lot of resources and not feasible for my local system , so for a workaround deployed with svm
- FastAPI is used to create a REST API for serving the SVM model.

### Docker 
- For FastAPI with docker https://fastapi.tiangolo.com/deployment/docker/ 
- Application can be packaged and deployed with any cloud service
- Build the container : docker build -t your-app-name
- Run the conatiner : docker run -p port:port your-app-name

### Deployment Steps
1. Install the required packages: `pip install fastapi scikit-learn uvicorn` or  `pip install -r requirements.txt`
2. Run the FastAPI app: `uvicorn app.main:app --reload`.
3. Make POST requests to the `/predict` endpoint with JSON payloads containing review text to get sentiment predictions.

### Example Screenshots
Below are example screenshots demonstrating the SVM model in action:
User Input:
![input](https://github.com/Soham7777/Bert-Sentiment/assets/66548809/b3517bf2-b208-4f1e-868c-fdbb447a7403)

Response from app:
![response](https://github.com/Soham7777/Bert-Sentiment/assets/66548809/8c8f1c37-c6bd-4f40-a6cb-7b678bf7e0c8)



## Getting Started

1. Clone this repository to your local machine.
2. Follow the deployment instructions above to run the FastAPI app.
3. Refer to the Jupyter Notebook for details on the deep learning approach.

