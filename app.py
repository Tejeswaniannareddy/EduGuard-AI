from flask import Flask, render_template, request
import joblib
import pandas as pd
import os


app = Flask(__name__)


MODEL_FILE = "model.pkl"


# ------------------------------------------------------------
# CHECK MODEL
# ------------------------------------------------------------

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        "model.pkl not found. "
        "Run train_model.py first."
    )


# ------------------------------------------------------------
# LOAD MODEL PACKAGE
# ------------------------------------------------------------

model_package = joblib.load(
    MODEL_FILE
)


model = model_package["model"]

FEATURES = model_package["features"]

THRESHOLD = model_package["threshold"]


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index2.html"
    )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------
        # Read form values
        # ----------------------------------------

        eye_gaze_angle = float(
            request.form.get(
                "eye_gaze_angle"
            )
        )


        audio_db = float(
            request.form.get(
                "audio_db"
            )
        )


        tab_switches = int(
            request.form.get(
                "tab_switches"
            )
        )


        gaze_rolling_10s = float(
            request.form.get(
                "gaze_rolling_10s"
            )
        )


        audio_rolling_10s = float(
            request.form.get(
                "audio_rolling_10s"
            )
        )


        # ----------------------------------------
        # Create input DataFrame
        # ----------------------------------------

        input_data = pd.DataFrame(
            [
                {
                    "eye_gaze_angle":
                        eye_gaze_angle,

                    "audio_db":
                        audio_db,

                    "tab_switches":
                        tab_switches,

                    "gaze_rolling_10s":
                        gaze_rolling_10s,

                    "audio_rolling_10s":
                        audio_rolling_10s
                }
            ]
        )


        # ----------------------------------------
        # Arrange features
        # ----------------------------------------

        input_data = input_data[
            FEATURES
        ]


        # ----------------------------------------
        # Get probability
        # ----------------------------------------

        probabilities = model.predict_proba(
            input_data
        )


        # Probability of class 1
        cheating_probability = (
            probabilities[0][1]
        )


        probability_percentage = (
            cheating_probability * 100
        )


        # ----------------------------------------
        # Apply 90% threshold
        # ----------------------------------------

        if cheating_probability >= THRESHOLD:

            result = "SUSPICIOUS"

        else:

            result = "NOT SUSPICIOUS"


        # ----------------------------------------
        # Return result
        # ----------------------------------------

        return render_template(
            "index2.html",

            result=result,

            probability=round(
                probability_percentage,
                2
            ),

            threshold=int(
                THRESHOLD * 100
            ),

            eye_gaze_angle=eye_gaze_angle,

            audio_db=audio_db,

            tab_switches=tab_switches,

            gaze_rolling_10s=gaze_rolling_10s,

            audio_rolling_10s=audio_rolling_10s
        )


    except Exception as error:

        return render_template(
            "index2.html",
            error=str(error)
        )


# ------------------------------------------------------------
# RUN APPLICATION
# ------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )