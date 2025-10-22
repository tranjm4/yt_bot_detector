"""
Model visualization logic for Streamlit app.

Displays anomaly detection results, user profiles, and model performance.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from db import (
    DatabaseConnection,
    get_latest_predictions,
    get_prediction_runs,
    get_user_prediction_history,
    get_user_comments,
    get_anomaly_trends,
)


@st.cache_resource
def get_db_connection() -> DatabaseConnection:
    """Initialize and cache database connection."""
    return DatabaseConnection()


def display_model_page():
    """Main model visualization page."""
    display_lessons_learned()
    display_methodology()

    st.header("Bot Detection Results")
    
    st.write("At the bottom of this section contains the top 100 users \
        based on anomaly scores from the isolation forest model. \
        You can play around and observe their comment histories. \
        Some of them are quite silly")

    try:
        db = get_db_connection()

        # Get predictions and model info
        predictions_df = get_latest_predictions(db, limit=100)

        if predictions_df.empty:
            st.warning("No predictions found. Run the pipeline first.")
            return

        latest_run = predictions_df.iloc[0]

        # Header with model info
        st.info(
            f"**Model:** {latest_run['modelname']} | "
            f"**Run Time:** {latest_run['runtimestamp']} | "
        )

        # Top row: Left = predictions table, Right = model visualizations
        col_left, col_right = st.columns([3, 2])

        with col_left:
            display_overview()
            st.divider()
            display_predictions_table(predictions_df)

        with col_right:
            display_model_visualizations_compact(latest_run['modelname'])

        # Bottom row: User details (full width)
        st.divider()
        display_user_details_section(db, predictions_df)

    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.info(
            "Make sure your database is running and environment variables are set:\n"
            "- POSTGRES_HOST\n"
            "- POSTGRES_PORT\n"
            "- POSTGRES_USER\n"
            "- POSTGRES_PASSWORD\n"
            "- POSTGRES_DB"
        )
    
def display_methodology():
    """Methodology text describing approach"""

    with st.expander(":blue[Methodology]"):
        st.markdown("""
        ## :blue[Methodology]
        
        ---
        ## :blue[Data]
        The data was collected from the [Google Youtube API](https://developers.google.com/youtube/v3/docs).
        
        The raw features I was able to extract can be detailed in my Postgres schema on my [GitHub repo](https://github.com/tranjm4/yt_bot_detector/blob/main/psql/schema.sql).
        
        ---
        
        Given that it's non-trivial to identify a bot based on
        manual inspection (and that we don't actually know the true labels), 
        it would be difficult to approach this problem as a
        supervised or even semi-supervised (i.e., manually labeling a few) task.
        
        Thus, I centered my approach around the following questions:
        - What temporal patterns should we expect?
        - What text content patterns should we expect?
        - How do we ensure the models we make are not bogus without ground truths?
        
        ### 1.  :blue[What temporal patterns should we expect?]
        - :red[**Deviation from peak hours** ]
            - Given that my data was centered around US political content, it's a reasonable assumption to make that
            the majority of the audience live within a 3-hour timezone difference. If a user is
            commenting opposite of peak hours (3am PST, 6am EST), it might raise some questions
            about the authenticity of the user (though we would like to take this with a grain of salt;
            US citizens living abroad, night shift workers, alternative lifestyles, etc., can be genuine factors)
        
        - :red[**Quick responses to videos**]
            - Additionally, we can observe patterns in how quickly users comment on videos.
            If a user comments within seconds on a majority of videos, it is likely that
        
        - :red[**Low variance in response delays**]
            - Similarly, if users are commenting, for example, exactly 20 minutes consistently
            after many videos (i.e., low variance in the delay), it might suggest some systematic behavior.
        
        ### 2. :blue[What text content patterns should we expect?]
        - :red[**Copy paste counts**]
            - We might expect bot users to be particularly 'spammy' with their activity. That is,
            frequent copy-pasting of comments may be deemed, at best, emotionally heated behavior and,
            at worst, bot behavior.
        - :red[**All-caps tendencies**]
            - We might be interested in the amount of all-caps behavior from a user's comment patterns.
            This could indicate a tendency to elicit strong emotional responses.
            
        ### 3. :blue[How do we evalute our model?]
        This is something that I found interesting as this is my first time working in an
        unsupervised learning setting. Obviously, the best way to verify our results
        is to manually check them. We can also run various checks to increase our confidence 
        in the performance in our model.
        - :green[**Manual inspection**]: This is the most obvious way, but it's not straight-forward.
        It can be time-consuming to check, since we don't have any ground truth labels.
        - :green[**Repeated isolation forest fittings**]: by running multiple isolation forests and
        comparing the average percentage of anomalies that are shared among the models 
        ([Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index)), we can ensure our
        isolation forest is robust to pure chance.
        - :green[**Anomaly score distribution**]: by checking for the counts of anomaly scores,
        we can visually determine how well our isolation forest is splitting from the rest of the data.
        - :green[**UMAP visualization**]: though not the strongest indicator, we can run it
        independently, adding the labels generated by the isolation forest afterwards
        to visualize how well the anomalies cluster in the data.
        """)
        
def display_lessons_learned():
    """Takeways text"""
    with st.expander(":blue[Lessons Learned]"):
        st.markdown("""
        This project was incredibly fun and challenging to work with, since this
        was my first time dealing with unsupervised models. Being able to gain
        insights on behavioral patterns felt very gratifying and empowering
        to know that these things are possible.
        
        I was also able to apply my working knowledge with MLOps ideas
        I've been learning (linked above), utilizing the following:
        - :blue[GitHub Actions] for continuous testing of my modules
        - Data versioning, feature engineering, and feature stores for more streamlined data pipelines
        - :blue[Docker containerization] for ease of end-to-end streamlining (data scraping -> feature engineering -> model training -> model evaluation)
        """)
        
def display_overview():
    """Overview text describing findings"""
    st.subheader("Results Overview")
    st.markdown("""
    The results of the model highlights some key points:
    - :orange[The isolation forest model appears to be somewhat stable]. Across
    random initializations, the model appears to have a 0.75 average Jaccard similarity.
    - :orange[The isolation forest model shows somewhat decent separation between
    anomalies and non-anomalies]. The :blue[Score Distribution] shows a cluster
    that's somewhat separated from non-anomaly users (it's shown in log-scale,
    which may appear a bit funky -- I plan on including non-log scalings soon).
    - :orange[The UMAP visualizations show a distinct separation from the
    rest of the data]. The 'spaghetti' patterns are likely attributed to the
    high-locality hyperparameters used to generate the model. It
    - :orange[Comment frequency and account age appears to be one of the more important features].
    Users that comment very often with relatively new accounts are being picked up as anomalies.
    """)

def display_predictions_table(predictions_df: pd.DataFrame):
    """Display predictions table."""
    st.subheader("Top Anomalous Users")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Number of users to display", 10, 100, 50, key="table_limit")
    with col2:
        sort_by = st.selectbox(
            "Sort by", ["Anomaly Score (Low to High)", "Account Age", "Comment Velocity"]
        )

    # Filter and sort
    display_df = predictions_df.head(limit).copy()

    # Parse feature values for display
    display_df["features"] = display_df["featurevalues"].apply(
        lambda x: x if isinstance(x, dict) else (json.loads(x) if isinstance(x, str) else {})
    )

    # Create display columns
    display_df["account_age_days"] = display_df["features"].apply(
        lambda x: x.get("account_age", 0)
    )
    display_df["comment_velocity"] = display_df["features"].apply(
        lambda x: x.get("comment_velocity", 0)
    )

    # Sort based on selection
    if sort_by == "Account Age":
        display_df = display_df.sort_values("account_age_days")
    elif sort_by == "Comment Velocity":
        display_df = display_df.sort_values("comment_velocity", ascending=False)

    # Display table
    st.dataframe(
        display_df[
            [
                "username",
                "anomalyscore",
                "accountcreatedate",
                "subcount",
                "videocount",
            ]
        ].rename(
            columns={
                "username": "Username",
                "anomalyscore": "Anomaly Score",
                "accountcreatedate": "Account Created",
                "subcount": "Subscribers",
                "videocount": "Videos",
            }
        ),
        width='stretch',
    )

    # Download button
    latest_run = predictions_df.iloc[0]
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"anomalous_users_{latest_run['runtimestamp']}.csv",
        mime="text/csv",
    )


def display_user_details_section(db: DatabaseConnection, predictions_df: pd.DataFrame):
    """Display detailed information for a specific user."""
    st.subheader("User Profile")

    # User selection
    user_options = {
        f"{row['username']} ({row['userid']})": row["userid"]
        for _, row in predictions_df.iterrows()
    }
    col1, col2 = st.columns(2)
    with col1:
        selected_user_key = st.selectbox("Select a user", list(user_options.keys()))
        selected_user_id = user_options[selected_user_key]

    # Get user data
    user_row = predictions_df[predictions_df["userid"] == selected_user_id].iloc[0]
    user_history = get_user_prediction_history(db, selected_user_id)
    user_comments = get_user_comments(db, selected_user_id, limit=50)

    # Display user profile
    with col2:
        st.metric("Anomaly Score", f"{user_row['anomalyscore']:.4f}")
        st.write(f"**Account Created:** {user_row['accountcreatedate']}")

    # Layout for feature values and prediction history
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Values")
        features = user_row["featurevalues"] if isinstance(user_row["featurevalues"], dict) else json.loads(user_row["featurevalues"])
        feature_df = pd.DataFrame(
            [{"Feature": k, "Value": v} for k, v in features.items()]
        )
        feature_df = feature_df.sort_values("Value", ascending=True)
        st.bar_chart(feature_df.set_index("Feature"), width='stretch', horizontal=True)

    with col2:
        if not user_history.empty:
            st.subheader("Prediction History")
            st.line_chart(
                user_history.set_index("runtimestamp")["anomalyscore"],
                width='stretch',
            )

    # Recent comments - full width
    if not user_comments.empty:
        st.subheader("Recent Comments")
        num_to_show = st.slider("Number of comments to show", 5, 50, 10, key="comment_slider")

        for _, comment in user_comments.head(num_to_show).iterrows():
            st.markdown(f"**{comment['publishdate']}** {comment['commenttext']}")
            st.caption(f"{comment['channelname']} • {comment['videotitle']} • {comment['likecount']} likes")
            st.divider()

        st.caption(f"Showing {min(num_to_show, len(user_comments))} of {len(user_comments)} comments")
    else:
        st.info("No comments found for this user.")


def display_model_visualizations_compact(model_name: str):
    """Display model performance visualizations."""
    st.subheader("Model Visualizations")

    # Get project root and results directory
    app_dir = Path(__file__).parent
    project_root = app_dir.parent
    results_dir = project_root / "results" / model_name

    if not results_dir.exists():
        st.warning(f"No visualizations found")
        return

    # Define visualization categories (prioritized order)
    viz_config = [
        ("UMAP", "umap_if_visualization.png"),
        ("Score Distribution", "anomaly_score_distribution.png"),
        ("Feature Importance", "permutation_importance.png"),
        ("Feature Differences", "feature_differences.png"),
        ("Ensemble Venn", "ensemble_venn.png"),
        ("Jaccard Similarity", "jaccard_similarity.png"),
    ]

    # Display all available visualizations stacked
    for viz_name, viz_file in viz_config:
        viz_path = results_dir / viz_file
        if viz_path.exists():
            st.markdown(f"**{viz_name}**")
            st.image(str(viz_path), width='stretch')