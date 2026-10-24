from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    """Load the saved fitted pipeline and its metadata."""
    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return pipeline, metadata


def main():
    st.set_page_config(
        page_title="Student Performance Prediction",
        page_icon="🎓",
    )

    st.title("Student Performance Prediction")
    st.write(
        "Enter study-related information to estimate a student's "
        "Pass or Fail result."
    )
    st.info(
        "This is an educational demonstration. Predictions should "
        "not be the sole basis for decisions about students."
    )

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error(
            "Model files are missing. Run the notebook's model-saving "
            "cells before starting this application."
        )
        st.stop()

    pipeline, metadata = load_artifacts()

    st.caption(
        f"Model: {metadata['model_name']} | "
        f"Labels were derived using an assumed pass threshold of "
        f"{metadata['pass_threshold']} out of 100."
    )

    with st.form("prediction_form"):
        hours_studied = st.number_input(
            "Study hours per week",
            min_value=0,
            max_value=168,
            value=20,
            step=1,
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0,
            max_value=100,
            value=80,
            step=1,
        )

        previous_scores = st.number_input(
            "Previous score (out of 100)",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
        )

        sleep_hours = st.number_input(
            "Sleep hours per day",
            min_value=0,
            max_value=24,
            value=7,
            step=1,
        )

        tutoring_sessions = st.number_input(
            "Number of tutoring sessions",
            min_value=0,
            value=1,
            step=1,
            help="Enter a non-negative whole number.",
        )

        access_to_resources = st.selectbox(
            "Access to learning resources",
            options=["Low", "Medium", "High"],
            index=1,
        )

        submitted = st.form_submit_button("Predict result")

    if submitted:
        # Check that study and sleep hours can fit within one week.
        if hours_studied + sleep_hours * 7 > 168:
            st.error(
                "Study hours and sleep hours together exceed "
                "the 168 hours available in a week."
            )
            st.stop()

        input_values = {
            "Hours_Studied": hours_studied,
            "Attendance": attendance,
            "Previous_Scores": previous_scores,
            "Sleep_Hours": sleep_hours,
            "Tutoring_Sessions": tutoring_sessions,
            "Access_to_Resources": access_to_resources,
        }

        # Match the feature names and order used during training.
        input_df = pd.DataFrame(
            [input_values],
            columns=metadata["feature_columns"],
        )

        # The saved pipeline performs preprocessing automatically.
        prediction = int(pipeline.predict(input_df)[0])
        result = metadata["class_labels"][str(prediction)]

        st.subheader(f"Predicted result: {result}")

        probabilities = pipeline.predict_proba(input_df)[0]
        fail_index = list(pipeline.classes_).index(metadata["positive_class"])
        fail_probability = float(probabilities[fail_index])

        st.metric(
            "Estimated probability of Fail",
            f"{fail_probability:.1%}",
        )
        st.caption(
            "This probability is a model estimate, not a guarantee. "
            "Its calibration has not been evaluated."
        )

        with st.expander("View submitted inputs"):
            st.dataframe(input_df, hide_index=True)

        # Download evidence of the actual interface output.
        result_record = {
            **input_values,
            "Predicted_Result": result,
            "Estimated_Fail_Probability": fail_probability,
            "Actual_Result": "Unknown",
        }

        st.download_button(
            label="Download prediction record",
            data=pd.DataFrame([result_record]).to_csv(index=False),
            file_name="prediction_record.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
