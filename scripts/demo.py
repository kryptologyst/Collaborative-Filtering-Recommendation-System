"""Streamlit demo for collaborative filtering recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorization, SVDModel
from src.data.dataset import MovieLensDataset
from src.evaluation.metrics import evaluate_model, compare_models


def load_data():
    """Load dataset and train models."""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = MovieLensDataset()
        st.session_state.dataset.load_data()
        
        # Split data
        train_data, test_data = st.session_state.dataset.split_data()
        st.session_state.train_data = train_data
        st.session_state.test_data = test_data
        
        # Train models
        with st.spinner("Training models..."):
            models = {
                'User-based CF': UserBasedCF(),
                'Item-based CF': ItemBasedCF(),
                'Matrix Factorization': MatrixFactorization(),
                'SVD': SVDModel()
            }
            
            trained_models = {}
            for name, model in models.items():
                with st.spinner(f"Training {name}..."):
                    model.fit(train_data)
                    trained_models[name] = model
            
            st.session_state.models = trained_models


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Collaborative Filtering Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Collaborative Filtering Recommendation System")
    st.markdown("Explore different collaborative filtering algorithms and their performance.")
    
    # Load data and models
    load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dataset Overview", "Model Comparison", "Recommendations", "Analysis"]
    )
    
    if page == "Dataset Overview":
        show_dataset_overview()
    elif page == "Model Comparison":
        show_model_comparison()
    elif page == "Recommendations":
        show_recommendations()
    elif page == "Analysis":
        show_analysis()


def show_dataset_overview():
    """Show dataset overview."""
    st.header("📊 Dataset Overview")
    
    dataset = st.session_state.dataset
    interactions = dataset.interactions
    items = dataset.items
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(interactions['user_id'].unique()))
    
    with col2:
        st.metric("Total Items", len(interactions['item_id'].unique()))
    
    with col3:
        st.metric("Total Interactions", len(interactions))
    
    with col4:
        sparsity = 1 - (len(interactions) / (len(interactions['user_id'].unique()) * len(interactions['item_id'].unique())))
        st.metric("Sparsity", f"{sparsity:.1%}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = interactions['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Rating Distribution",
        labels={'x': 'Rating', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # User activity
    st.subheader("User Activity")
    user_stats = dataset.get_user_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            user_stats,
            x='n_items',
            title="Items per User",
            labels={'n_items': 'Number of Items', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            user_stats,
            x='avg_rating',
            title="Average Rating per User",
            labels={'avg_rating': 'Average Rating', 'count': 'Number of Users'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Item popularity
    st.subheader("Item Popularity")
    item_stats = dataset.get_item_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            item_stats,
            x='n_users',
            title="Users per Item",
            labels={'n_users': 'Number of Users', 'count': 'Number of Items'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            item_stats,
            x='avg_rating',
            title="Average Rating per Item",
            labels={'avg_rating': 'Average Rating', 'count': 'Number of Items'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_model_comparison():
    """Show model comparison."""
    st.header("🏆 Model Comparison")
    
    # Evaluate models
    with st.spinner("Evaluating models..."):
        models = st.session_state.models
        test_data = st.session_state.test_data
        
        comparison_df = compare_models(models, test_data)
    
    # Display results
    st.subheader("Performance Metrics")
    
    # Format the dataframe for better display
    display_df = comparison_df.copy()
    for col in display_df.columns:
        if col != 'model' and isinstance(display_df[col].iloc[0], float):
            display_df[col] = display_df[col].round(4)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Visualization
    st.subheader("Performance Visualization")
    
    # Select metrics to plot
    metric_cols = [col for col in comparison_df.columns if col != 'model']
    selected_metrics = st.multiselect(
        "Select metrics to visualize",
        metric_cols,
        default=metric_cols[:4]
    )
    
    if selected_metrics:
        # Create subplots
        fig = make_subplots(
            rows=len(selected_metrics),
            cols=1,
            subplot_titles=selected_metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(
                go.Bar(
                    x=comparison_df['model'],
                    y=comparison_df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=i+1,
                col=1
            )
        
        fig.update_layout(
            height=200 * len(selected_metrics),
            title_text="Model Performance Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_recommendations():
    """Show recommendations interface."""
    st.header("🎯 Get Recommendations")
    
    dataset = st.session_state.dataset
    models = st.session_state.models
    
    # User selection
    user_id = st.selectbox(
        "Select a user",
        sorted(dataset.interactions['user_id'].unique())
    )
    
    # Model selection
    model_name = st.selectbox(
        "Select a model",
        list(models.keys())
    )
    
    model = models[model_name]
    
    # Number of recommendations
    n_recs = st.slider("Number of recommendations", 5, 50, 10)
    
    # Get recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            recommendations = model.recommend(user_id, n_recs)
        
        if recommendations:
            st.subheader(f"Top {len(recommendations)} Recommendations for User {user_id}")
            
            # Create recommendations dataframe
            rec_df = pd.DataFrame(recommendations, columns=['item_id', 'score'])
            
            # Add item information if available
            if dataset.items is not None:
                rec_df = rec_df.merge(dataset.items, on='item_id', how='left')
                rec_df = rec_df[['item_id', 'title', 'genres', 'score']]
            else:
                rec_df['title'] = f"Item {rec_df['item_id']}"
                rec_df['genres'] = "Unknown"
            
            # Display recommendations
            for i, (_, row) in enumerate(rec_df.iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**#{i}**")
                    
                    with col2:
                        st.write(f"**{row['title']}**")
                        st.write(f"Genres: {row['genres']}")
                    
                    with col3:
                        st.write(f"Score: {row['score']:.3f}")
                    
                    st.divider()
        else:
            st.warning("No recommendations available for this user.")
    
    # Show user's history
    st.subheader("User's Rating History")
    user_interactions = dataset.interactions[dataset.interactions['user_id'] == user_id]
    
    if not user_interactions.empty:
        # Add item information if available
        if dataset.items is not None:
            user_history = user_interactions.merge(dataset.items, on='item_id', how='left')
            user_history = user_history[['item_id', 'title', 'genres', 'rating', 'timestamp']]
        else:
            user_history = user_interactions.copy()
            user_history['title'] = user_history['item_id'].apply(lambda x: f"Item {x}")
            user_history['genres'] = "Unknown"
        
        # Sort by timestamp
        user_history = user_history.sort_values('timestamp', ascending=False)
        
        st.dataframe(user_history, use_container_width=True)
    else:
        st.info("No rating history available for this user.")


def show_analysis():
    """Show analysis and insights."""
    st.header("📈 Analysis & Insights")
    
    dataset = st.session_state.dataset
    interactions = dataset.interactions
    
    # Temporal analysis
    st.subheader("Temporal Analysis")
    
    # Convert timestamp to datetime
    interactions['date'] = pd.to_datetime(interactions['timestamp'], unit='s')
    interactions['month'] = interactions['date'].dt.to_period('M')
    
    # Monthly activity
    monthly_activity = interactions.groupby('month').size().reset_index(name='interactions')
    monthly_activity['month'] = monthly_activity['month'].astype(str)
    
    fig = px.line(
        monthly_activity,
        x='month',
        y='interactions',
        title="Monthly Activity",
        labels={'month': 'Month', 'interactions': 'Number of Interactions'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating trends
    monthly_ratings = interactions.groupby('month')['rating'].mean().reset_index()
    monthly_ratings['month'] = monthly_ratings['month'].astype(str)
    
    fig = px.line(
        monthly_ratings,
        x='month',
        y='rating',
        title="Average Rating Over Time",
        labels={'month': 'Month', 'rating': 'Average Rating'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre analysis (if available)
    if dataset.items is not None:
        st.subheader("Genre Analysis")
        
        # Extract genres
        items = dataset.items.copy()
        items['genres'] = items['genres'].str.split('|')
        items = items.explode('genres')
        
        # Genre popularity
        genre_popularity = items['genres'].value_counts()
        
        fig = px.pie(
            values=genre_popularity.values,
            names=genre_popularity.index,
            title="Genre Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre ratings
        genre_ratings = interactions.merge(items, on='item_id').groupby('genres')['rating'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=genre_ratings.index,
            y=genre_ratings.values,
            title="Average Rating by Genre",
            labels={'x': 'Genre', 'y': 'Average Rating'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
