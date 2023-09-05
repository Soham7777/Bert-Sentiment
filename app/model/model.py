import pickle
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

__version__ = "1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/amazon_reviews_svm_model.pkl", "rb") as f:
    model = pickle.load(f)  

# #Now you can use the loaded_model to make predictions
# predictions = model.predict(['Not recommended'])
# print(f'The predicted value is {predictions}')



def predict_pipeline(text):
    # text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    # text = re.sub(r"[[]]", " ", text)
    #text = text.lower()
    pred = model.predict([text])
    result = "Negative review" if pred == 0 else "Positive review"
    return result