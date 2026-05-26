import skops.io as sio
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
from utils import data_prep
# 1. Initialize the FastAPI app
app = FastAPI(
    title="User Prediction Service",
    description=(
        "CTG demo API that predicts outcomes"
        "based on username features."
    ),
    version="0.0.1"
)

# --- ML Model & Data Setup Placeholder ---
# In your real code, you would load your model and data here:
# import joblib
# import pandas as pd
MODEL = sio.load("model.skops", trusted=[])
DATA, COUNTRY_MAP = data_prep()


def get_predict(dataframe: pd.DataFrame) -> str:
    inputs = dataframe.drop(['id', 'country_destination'], axis=1)
    pred = MODEL.predict(inputs)
    return COUNTRY_MAP[pred.item()]
# ----------------------------------------


class PredictionRequest(BaseModel):
    username: str = Field(
        ...,
        description="The unique username to generate a prediction for",
        min_length=1,
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"username": "exampleusername"}
        }
    )


class PredictionResponse(BaseModel):
    username: str
    prediction: str
    status: str = "success"


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a prediction for a specific user"
)
async def get_prediction(payload: PredictionRequest):
    """
    Takes a username, looks up their features from the database,
    and passes them to the scikit-learn model to return a prediction.
    """
    # Standardize input (e.g., lowercase) to prevent casing mismatches
    target_username = payload.username.lower().strip()

    # Fetch user features (Replace with your actual pandas/DB lookup)
    user_features = DATA[DATA['id'] == target_username]
    if user_features.shape[0] == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{payload.username}' not found in the dataset."
        )
    try:
        prediction_result = get_predict(user_features)
    except Exception as e:
        # Catch unexpected model failures gracefully
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}"
        )

    # Return the validated response
    return PredictionResponse(
        username=target_username,
        prediction=prediction_result)
