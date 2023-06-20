"""
THIS IS JUST A TEMPLATE - CHANGE IT TO FIT YOUR NEEDS
"""

import joblib
import bentoml


with open("/userRepoData/__sidetrek__/taeefnajib/Diamond-Price-Prediction/bentoml/models/c173172ab4338526eb93aec1ce3d2a9a", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.lightgbm.save_model(
        "diamond_model",
        model,
    )
    print(saved_model) # This is required!