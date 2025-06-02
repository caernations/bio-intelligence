import os
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Bio Intelligence Dashboard", layout="wide", page_icon="🧬")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Enhanced color scheme for biology
theme_colors = {
    "primary": "#2E8B57",      # Sea Green
    "secondary": "#DAA520",    # Goldenrod
    "success": "#228B22",      # Forest Green
    "warning": "#FF8C00",      # Dark Orange
    "danger": "#DC143C",       # Crimson
    "neutral": "#708090",      # Slate Gray
    "plant": "#228B22",        # Forest Green
    "animal": "#4169E1",       # Royal Blue
    "biodiversity": "#9932CC"  # Dark Orchid
}

@st.cache_data
def load_biological_data():
    """Load all biological datasets"""
    try:
        # Load animal data
        animal_files = [f"bio_dataset/animal/animal{i}_cleaned.csv" for i in range(1, 7)]
        animal_dfs = []
        for file in animal_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                df['dataset_source'] = file.split('/')[-1]
                animal_dfs.append(df)
        animals_df = pd.concat(animal_dfs, ignore_index=True) if animal_dfs else pd.DataFrame()
        
        # Load plant data
        plants_df = pd.read_csv("bio_dataset/plant/plant1_cleaned.csv") if os.path.exists("bio_dataset/plant/plant1_cleaned.csv") else pd.DataFrame()
        
        # Load biodiversity data
        biodiversity_files = [f"bio_dataset/biodiversity/biodiversity{i}_cleaned.csv" for i in range(1, 3)]
        biodiversity_dfs = []
        for file in biodiversity_files:
            if os.path.exists(file):
                try:
                    df = pd.read_csv(file)
                    df['dataset_source'] = file.split('/')[-1]
                    biodiversity_dfs.append(df)
                except Exception as e:
                    st.warning(f"Could not load {file}: {e}")
        biodiversity_df = pd.concat(biodiversity_dfs, ignore_index=True) if biodiversity_dfs else pd.DataFrame()
        
        return animals_df, plants_df, biodiversity_df
    except Exception as e:
        st.error(f"Error loading biological data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def interpret_bio_query(query):
    """Interpret natural language queries about biological data"""
    q = query.lower()
    
    categories = {
        "animal_taxonomy": ["animal", "species", "genus", "family", "order", "class", "phylum", 
                           "taxonomy", "taxonomic", "classification", "mammals", "birds", "fish",
                           "reptiles", "amphibians", "insects", "vertebrates", "invertebrates"],
        
        "plant_characteristics": ["plant", "flora", "flower", "bloom", "leaf", "tree", "shrub", 
                                "herb", "growth", "height", "color", "soil", "light", "habitat",
                                "perennial", "annual", "deciduous", "evergreen"],
        
        "biodiversity_patterns": ["biodiversity", "diversity", "richness", "abundance", "distribution",
                                "ecosystem", "habitat", "conservation", "endemic", "native",
                                "population", "community", "biome"],
        
        "ecological_analysis": ["ecology", "environment", "adaptation", "evolution", "genetics",
                               "behavior", "interaction", "predator", "prey", "symbiosis",
                               "migration", "reproduction", "lifecycle"],
        
        "comparison_analysis": ["compare", "comparison", "versus", "vs", "difference", "similar",
                               "related", "relationship", "correlation"],
        
        "geographic_analysis": ["geographic", "location", "region", "continent", "country",
                               "climate", "temperature", "precipitation", "altitude"],
        
        "morphological_analysis": ["size", "weight", "length", "structure", "anatomy", "morphology",
                                  "physical", "characteristics", "features"],
        
        "conservation_status": ["endangered", "threatened", "extinct", "conservation", "protected",
                               "rare", "vulnerable", "status", "iucn", "red list"]
    }
    
    for category, keywords in categories.items():
        if any(word in q for word in keywords):
            return category
    
    # Default category based on dataset content
    if any(word in q for word in ["kingdom", "phylum", "class"]):
        return "animal_taxonomy"
    elif any(word in q for word in ["scientific_name", "common_name", "family"]):
        return "plant_characteristics"
    else:
        return "general_biology"

def filter_biological_data(keyword, query=None, animals_df=None, plants_df=None, biodiversity_df=None):
    """Filter and analyze biological data based on query type"""
    
    if keyword == "animal_taxonomy":
        if animals_df.empty:
            return pd.DataFrame(), None, None, "No animal data available", None
        
        # Analyze taxonomic distribution
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        available_levels = [level for level in taxonomy_levels if level in animals_df.columns]
        
        if available_levels:
            # Count species by different taxonomic levels
            analysis_data = []
            for level in available_levels[:4]:  # Top 4 levels
                counts = animals_df[level].value_counts().head(10)
                for category, count in counts.items():
                    analysis_data.append({
                        'taxonomic_level': level.title(),
                        'category': category,
                        'count': count
                    })
            
            result_df = pd.DataFrame(analysis_data)
            
            # Additional analysis: Class distribution
            if 'class' in animals_df.columns:
                class_dist = animals_df['class'].value_counts().head(15).reset_index()
                class_dist.columns = ['class', 'species_count']
                
                return result_df, "taxonomic_level", "count", "Animal Taxonomic Analysis", class_dist
        
        return animals_df.head(20), None, None, "Animal Species Overview", None
    
    elif keyword == "plant_characteristics":
        if plants_df.empty:
            return pd.DataFrame(), None, None, "No plant data available", None
        
        # Analyze plant characteristics
        analysis_data = []
        
        # Family distribution
        if 'family' in plants_df.columns:
            family_counts = plants_df['family'].value_counts().head(10)
            
        # Growth habit analysis
        if 'habit' in plants_df.columns:
            habit_counts = plants_df['habit'].value_counts()
            
        # Bloom period analysis
        if 'bloom_period' in plants_df.columns:
            bloom_analysis = plants_df['bloom_period'].value_counts().head(10)
        
        # Light requirements
        if 'light' in plants_df.columns:
            light_req = plants_df['light'].value_counts()
            
        # Create summary dataframe
        summary_data = []
        
        if 'family' in plants_df.columns:
            for family, count in family_counts.items():
                summary_data.append({'category': 'Family', 'subcategory': family, 'count': count})
        
        if 'habit' in plants_df.columns:
            for habit, count in habit_counts.items():
                summary_data.append({'category': 'Growth Habit', 'subcategory': habit, 'count': count})
                
        summary_df = pd.DataFrame(summary_data)
        
        # Additional data for charts
        additional_data = {
            'family_dist': family_counts.reset_index() if 'family' in plants_df.columns else None,
            'habit_dist': habit_counts.reset_index() if 'habit' in plants_df.columns else None,
            'light_req': light_req.reset_index() if 'light' in plants_df.columns else None
        }
        
        return summary_df, "category", "count", "Plant Characteristics Analysis", additional_data
    
    elif keyword == "biodiversity_patterns":
        if biodiversity_df.empty:
            return pd.DataFrame(), None, None, "No biodiversity data available", None
        
        # Analyze biodiversity patterns
        numeric_cols = biodiversity_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Create summary statistics
            summary_stats = biodiversity_df[numeric_cols].describe().round(2)
            summary_df = summary_stats.T.reset_index()
            summary_df.rename(columns={'index': 'metric'}, inplace=True)
            
            return summary_df, "metric", "mean", "Biodiversity Metrics Summary", None
        
        return biodiversity_df.head(20), None, None, "Biodiversity Data Overview", None
    
    elif keyword == "comparison_analysis":
        # Compare different biological datasets
        comparison_data = []
        
        if not animals_df.empty:
            comparison_data.append({
                'dataset': 'Animals',
                'total_records': len(animals_df),
                'unique_families': animals_df['family'].nunique() if 'family' in animals_df.columns else 0,
                'data_type': 'Taxonomic'
            })
        
        if not plants_df.empty:
            comparison_data.append({
                'dataset': 'Plants',
                'total_records': len(plants_df),
                'unique_families': plants_df['family'].nunique() if 'family' in plants_df.columns else 0,
                'data_type': 'Ecological'
            })
        
        if not biodiversity_df.empty:
            comparison_data.append({
                'dataset': 'Biodiversity',
                'total_records': len(biodiversity_df),
                'unique_families': 0,  # Will be calculated based on actual columns
                'data_type': 'Community'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df, "dataset", "total_records", "Dataset Comparison", None
    
    else:
        # General overview
        overview_data = []
        
        if not animals_df.empty:
            overview_data.append({
                'category': 'Animal Species',
                'count': len(animals_df),
                'unique_families': animals_df['family'].nunique() if 'family' in animals_df.columns else 0
            })
        
        if not plants_df.empty:
            overview_data.append({
                'category': 'Plant Species',
                'count': len(plants_df),
                'unique_families': plants_df['family'].nunique() if 'family' in plants_df.columns else 0
            })
        
        if not biodiversity_df.empty:
            overview_data.append({
                'category': 'Biodiversity Records',
                'count': len(biodiversity_df),
                'unique_families': 0
            })
        
        overview_df = pd.DataFrame(overview_data)
        return overview_df, "category", "count", "Biological Data Overview", None

def create_biological_visualization(df, plot_type, x_col, y_col, title=None, additional_data=None, category="general"):
    """Create visualizations for biological data"""
    try:
        # Choose colors based on biological category
        if category == "plant":
            color_palette = [theme_colors["plant"], theme_colors["success"], "#90EE90"]
        elif category == "animal":
            color_palette = [theme_colors["animal"], "#87CEEB", "#4682B4"]
        else:
            color_palette = [theme_colors["primary"], theme_colors["secondary"], theme_colors["biodiversity"]]
        
        if plot_type == "bar":
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                color_discrete_sequence=color_palette
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=50, b=100),
                height=500
            )
            
        elif plot_type == "sunburst":
            # For taxonomic hierarchy visualization
            if 'taxonomic_level' in df.columns and 'category' in df.columns:
                fig = px.sunburst(
                    df, path=['taxonomic_level', 'category'], values='count',
                    title=title or "Taxonomic Hierarchy",
                    color_discrete_sequence=color_palette
                )
            else:
                return None
                
        elif plot_type == "treemap":
            if len(df.columns) >= 3:
                fig = px.treemap(
                    df, path=[x_col], values=y_col,
                    title=title or f"Distribution of {y_col.replace('_', ' ').title()}",
                    color_discrete_sequence=color_palette
                )
            else:
                return None
                
        elif plot_type == "scatter":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = px.scatter(
                    df, x=numeric_cols[0], y=numeric_cols[1],
                    title=title or f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                    color_discrete_sequence=color_palette,
                    hover_data=df.columns.tolist()[:5]
                )
            else:
                return None
                
        elif plot_type == "pie":
            fig = px.pie(
                df, names=x_col, values=y_col,
                title=title or f"Distribution of {y_col.replace('_', ' ').title()}",
                color_discrete_sequence=color_palette
            )
            
        elif plot_type == "heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    title=title or "Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
            else:
                return None
                
        elif plot_type == "box":
            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                fig = px.box(
                    df, x=x_col, y=y_col,
                    title=title or f"Distribution of {y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                    color_discrete_sequence=color_palette
                )
            else:
                return None
                
        else:
            # Default to bar chart
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                color_discrete_sequence=color_palette
            )
            
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def generate_biological_insights(keyword, df, x_col, y_col, additional_data):
    """Generate additional charts and insights for biological data"""
    charts = []
    
    if keyword == "animal_taxonomy":
        # Create phylogenetic diversity chart
        if additional_data is not None and 'class' in additional_data.columns:
            class_fig = px.bar(
                additional_data, x="class", y="species_count",
                title="Species Count by Animal Class",
                labels={"class": "Animal Class", "species_count": "Number of Species"},
                color_discrete_sequence=[theme_colors["animal"]]
            )
            class_fig.update_layout(xaxis_tickangle=-45)
            charts.append(("Species by Animal Class", class_fig))
        
        # Create taxonomic level distribution
        if 'taxonomic_level' in df.columns:
            level_dist = df.groupby('taxonomic_level')['count'].sum().reset_index()
            level_fig = px.pie(
                level_dist, names='taxonomic_level', values='count',
                title="Distribution Across Taxonomic Levels",
                color_discrete_sequence=[theme_colors["animal"], theme_colors["secondary"], theme_colors["success"]]
            )
            charts.append(("Taxonomic Level Distribution", level_fig))
    
    elif keyword == "plant_characteristics":
        if additional_data:
            # Family distribution
            if additional_data.get('family_dist') is not None:
                family_fig = px.bar(
                    additional_data['family_dist'], x='family', y='count',
                    title="Plant Species by Family",
                    labels={"family": "Plant Family", "count": "Number of Species"},
                    color_discrete_sequence=[theme_colors["plant"]]
                )
                family_fig.update_layout(xaxis_tickangle=-45)
                charts.append(("Species by Plant Family", family_fig))
            
            # Growth habit distribution
            if additional_data.get('habit_dist') is not None:
                habit_fig = px.pie(
                    additional_data['habit_dist'], names='habit', values='count',
                    title="Distribution by Growth Habit",
                    color_discrete_sequence=[theme_colors["plant"], theme_colors["success"], "#90EE90", "#228B22"]
                )
                charts.append(("Growth Habit Distribution", habit_fig))
            
            # Light requirements
            if additional_data.get('light_req') is not None:
                light_fig = px.bar(
                    additional_data['light_req'], x='light', y='count',
                    title="Plants by Light Requirements",
                    labels={"light": "Light Requirement", "count": "Number of Species"},
                    color_discrete_sequence=[theme_colors["warning"]]
                )
                charts.append(("Light Requirements", light_fig))
    
    elif keyword == "biodiversity_patterns":
        # Create biodiversity metrics visualization
        if 'mean' in df.columns and 'std' in df.columns:
            metrics_fig = px.scatter(
                df, x='mean', y='std', hover_name='metric',
                title="Biodiversity Metrics: Mean vs Standard Deviation",
                labels={"mean": "Mean Value", "std": "Standard Deviation"},
                color_discrete_sequence=[theme_colors["biodiversity"]]
            )
            charts.append(("Metrics Scatter Plot", metrics_fig))
    
    return charts

def build_biology_dashboard():
    """Main dashboard interface"""
    st.title("🧬 Bio Intelligence Dashboard")
    st.markdown("### Comprehensive Analysis of Biological Data")
    
    # Load data
    with st.spinner("Loading biological datasets..."):
        animals_df, plants_df, biodiversity_df = load_biological_data()
    
    # Sidebar for data overview
    with st.sidebar:
        st.header("📊 Data Overview")
        if not animals_df.empty:
            st.metric("Animal Species", len(animals_df))
        if not plants_df.empty:
            st.metric("Plant Species", len(plants_df))
        if not biodiversity_df.empty:
            st.metric("Biodiversity Records", len(biodiversity_df))
        
        st.divider()
        st.header("🎯 Quick Filters")
        dataset_filter = st.selectbox(
            "Focus on Dataset:",
            ["All Datasets", "Animals Only", "Plants Only", "Biodiversity Only"]
        )
    
    # Main tabs
    main_tabs = st.tabs(["🔍 Natural Language Analysis", "📈 Data Explorer", "🌿 Comparative Analysis"])
    
    with main_tabs[0]:
        st.header("Natural Language Biological Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_method = st.radio(
                "Analysis Method:", 
                ["Natural Language Query", "Guided Analysis"], 
                horizontal=True
            )
            
            if search_method == "Natural Language Query":
                user_query = st.text_input(
                    "Ask about biological data:",
                    placeholder="e.g., 'What animals are in the Chordata phylum?' or 'Show me flowering plants by family'"
                )
                st.info("💡 Try asking about animal taxonomy, plant characteristics, or biodiversity patterns!")
                
            else:
                analysis_type = st.selectbox("Analysis Type:", [
                    "Animal Taxonomy", "Plant Characteristics", "Biodiversity Patterns",
                    "Ecological Analysis", "Comparison Analysis", "Conservation Status"
                ])
                
                if analysis_type == "Animal Taxonomy":
                    if not animals_df.empty and 'class' in animals_df.columns:
                        selected_classes = st.multiselect("Animal Classes:", animals_df['class'].unique())
                        taxonomic_level = st.selectbox("Focus Level:", ["All", "Family", "Order", "Class"])
                        user_query = f"Show animal taxonomy for {', '.join(selected_classes) if selected_classes else 'all classes'} at {taxonomic_level} level"
                    else:
                        user_query = "Show animal taxonomy overview"
                        
                elif analysis_type == "Plant Characteristics":
                    if not plants_df.empty:
                        if 'habit' in plants_df.columns:
                            selected_habits = st.multiselect("Growth Habits:", plants_df['habit'].unique())
                        if 'light' in plants_df.columns:
                            light_pref = st.selectbox("Light Preference:", ["All"] + list(plants_df['light'].unique()))
                        user_query = f"Analyze plant characteristics for selected criteria"
                    else:
                        user_query = "Show plant characteristics overview"
                        
                else:
                    user_query = f"Analyze {analysis_type.lower()}"
        
        with col2:
            show_raw_data = st.checkbox("Show Raw Data", value=True)
            enable_ai_analysis = st.checkbox("AI Insights", value=True)
            
            st.divider()
            chart_type = st.selectbox("Visualization:", [
                "bar", "pie", "sunburst", "treemap", "scatter", "box", "heatmap"
            ])
            
            color_scheme = st.selectbox("Color Theme:", [
                "Biological", "Forest", "Ocean", "Autumn"
            ])
            
            if color_scheme == "Forest":
                theme_colors["primary"] = "#228B22"
            elif color_scheme == "Ocean":
                theme_colors["primary"] = "#4169E1"
            elif color_scheme == "Autumn":
                theme_colors["primary"] = "#DAA520"
        
        if 'user_query' in locals() and user_query:
            st.info(f"🔬 Analyzing: **{user_query}**")
            
            with st.spinner("Processing biological data..."):
                keyword = interpret_bio_query(user_query)
                result = filter_biological_data(keyword, user_query, animals_df, plants_df, biodiversity_df)
                
                if len(result) == 5:
                    df, x_col, y_col, title, additional_data = result
                else:
                    df, x_col, y_col, title = result
                    additional_data = None
                
                # Determine biological category for coloring
                bio_category = "plant" if "plant" in keyword else "animal" if "animal" in keyword else "general"
                
                analysis_tabs = st.tabs(["📊 Visualizations", "🧠 Insights", "📋 Data Details", "🤖 AI Analysis"])
                
                with analysis_tabs[0]:
                    st.subheader(title)
                    
                    if x_col and y_col and not df.empty:
                        chart_col1, chart_col2 = st.columns([3, 1])
                        
                        with chart_col2:
                            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                                st.metric("Total Count", f"{df[y_col].sum():,}")
                                st.metric("Average", f"{df[y_col].mean():.1f}")
                                st.metric("Maximum", f"{df[y_col].max():,}")
                                
                                if len(df) > 1:
                                    st.metric("Diversity Index", f"{len(df):,}")
                        
                        with chart_col1:
                            fig = create_biological_visualization(
                                df, chart_type, x_col, y_col, title, additional_data, bio_category
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Unable to create selected visualization with current data")
                    else:
                        st.dataframe(df, use_container_width=True)
                
                with analysis_tabs[1]:
                    st.subheader("🔬 Biological Insights")
                    
                    if x_col and y_col:
                        charts = generate_biological_insights(keyword, df, x_col, y_col, additional_data)
                        
                        if charts:
                            for i, (chart_title, chart_fig) in enumerate(charts):
                                if i % 2 == 0:
                                    cols = st.columns(2)
                                
                                with cols[i % 2]:
                                    st.subheader(chart_title)
                                    st.plotly_chart(chart_fig, use_container_width=True)
                        else:
                            st.info("No additional insights available for this query type")
                    else:
                        st.info("Select a query type that generates insights")
                
                with analysis_tabs[2]:
                    st.subheader("📋 Detailed Data")
                    
                    if show_raw_data and not df.empty:
                        # Data filtering and pagination
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            rows_per_page = st.slider("Rows per page", 5, 50, 20)
                        with col2:
                            page = st.number_input(
                                "Page", 
                                min_value=1, 
                                max_value=max(1, len(df) // rows_per_page + 1), 
                                step=1
                            )
                        with col3:
                            search_term = st.text_input("Search in data", "")
                        
                        # Apply search filter
                        filtered_df = df
                        if search_term:
                            text_cols = df.select_dtypes(include=['object']).columns
                            mask = df[text_cols].astype(str).apply(
                                lambda x: x.str.contains(search_term, case=False, na=False)
                            ).any(axis=1)
                            filtered_df = df[mask]
                        
                        # Pagination
                        start_idx = (page - 1) * rows_per_page
                        end_idx = min(start_idx + rows_per_page, len(filtered_df))
                        
                        st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
                        st.write(f"Showing {start_idx+1} to {end_idx} of {len(filtered_df)} entries")
                        
                        # Data summary
                        if not filtered_df.empty:
                            st.subheader("Data Summary")
                            summary_col1, summary_col2 = st.columns(2)
                            
                            with summary_col1:
                                st.write("**Column Information:**")
                                col_info = pd.DataFrame({
                                    'Column': filtered_df.columns,
                                    'Type': [str(dtype) for dtype in filtered_df.dtypes],
                                    'Non-Null': filtered_df.count(),
                                    'Unique': [filtered_df[col].nunique() for col in filtered_df.columns]
                                })
                                st.dataframe(col_info, use_container_width=True)
                            
                            with summary_col2:
                                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
                                if numeric_cols:
                                    st.write("**Numeric Statistics:**")
                                    st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
                
                with analysis_tabs[3]:
                    st.subheader("🤖 AI-Powered Analysis")
                    
                    if enable_ai_analysis and client:
                        # Prepare context for AI analysis
                        context = f"Biological Analysis Type: {keyword}\n"
                        context += f"Dataset: {title}\n"
                        context += f"Data Summary:\n{df.describe().to_string() if not df.empty else 'No data available'}\n\n"
                        
                        if keyword == "animal_taxonomy":
                            context += "Focus: Animal taxonomic classification and diversity\n"
                            if not animals_df.empty:
                                context += f"Total animal species: {len(animals_df)}\n"
                                if 'class' in animals_df.columns:
                                    top_classes = animals_df['class'].value_counts().head(5)
                                    context += f"Top animal classes: {top_classes.to_string()}\n"
                        
                        elif keyword == "plant_characteristics":
                            context += "Focus: Plant characteristics and ecological requirements\n"
                            if not plants_df.empty:
                                context += f"Total plant species: {len(plants_df)}\n"
                                if 'family' in plants_df.columns:
                                    top_families = plants_df['family'].value_counts().head(5)
                                    context += f"Top plant families: {top_families.to_string()}\n"
                        
                        elif keyword == "biodiversity_patterns":
                            context += "Focus: Biodiversity patterns and ecological metrics\n"
                            if not biodiversity_df.empty:
                                context += f"Biodiversity records: {len(biodiversity_df)}\n"
                        
                        prompt = f"""Biological Data Analysis Context:
                        {context}
                        
                        User Query: {user_query}
                        
                        Please provide:
                        1. Key biological insights from the data (3-4 bullet points)
                        2. Ecological significance and patterns
                        3. Conservation or research implications
                        4. Recommendations for further analysis
                        
                        Format your response in clear sections with headers. Focus on biological accuracy and scientific interpretation. Respond in Indonesian language."""
                        
                        with st.spinner("Generating AI analysis..."):
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "Anda adalah seorang ahli biologi dan ekologi yang memberikan analisis mendalam tentang data biologis. Berikan wawasan ilmiah yang akurat dengan fokus pada signifikansi ekologi dan konservasi. Selalu berikan respons dalam Bahasa Indonesia."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1000,
                                    temperature=0.3,
                                )
                                answer = response.choices[0].message.content
                                st.markdown(answer)
                            except Exception as e:
                                st.error(f"Error generating AI analysis: {str(e)}")
                                st.markdown("""
                                ## Analisis Dasar Biologis
                                
                                Berdasarkan data yang tersedia, berikut beberapa observasi:
                                
                                * **Keanekaragaman**: Data menunjukkan pola keanekaragaman yang menarik untuk investigasi lebih lanjut
                                * **Distribusi**: Pola distribusi taksonomi atau ekologi perlu diperiksa lebih mendalam
                                * **Karakteristik**: Identifikasi karakteristik unik dari kelompok organisme yang dianalisis
                                * **Konservasi**: Pertimbangkan implikasi konservasi dari pola yang diamati
                                
                                Aktifkan API OpenAI untuk analisis AI yang lebih detail.
                                """)
                    else:
                        st.info("Aktifkan Analisis AI untuk mendapatkan wawasan biologis mendalam")
                        
                        # Basic biological insights without AI
                        if not df.empty:
                            st.write("**Ringkasan Biologis:**")
                            
                            insights_col1, insights_col2 = st.columns(2)
                            
                            with insights_col1:
                                if 'count' in df.columns:
                                    total_records = df['count'].sum() if df['count'].dtype in [np.int64, np.float64] else len(df)
                                    st.metric("Total Records", f"{total_records:,}")
                                
                                if len(df) > 1:
                                    st.metric("Diversity Categories", len(df))
                            
                            with insights_col2:
                                if x_col and x_col in df.columns:
                                    unique_categories = df[x_col].nunique()
                                    st.metric(f"Unique {x_col.replace('_', ' ').title()}", unique_categories)
                                
                                if y_col and y_col in df.columns and df[y_col].dtype in [np.int64, np.float64]:
                                    avg_value = df[y_col].mean()
                                    st.metric(f"Average {y_col.replace('_', ' ').title()}", f"{avg_value:.1f}")
    
    with main_tabs[1]:
        st.header("📈 Biological Data Explorer")
        
        # Dataset selection
        dataset_tabs = st.tabs(["🐾 Animals", "🌱 Plants", "🌍 Biodiversity"])
        
        with dataset_tabs[0]:
            st.subheader("Animal Dataset Explorer")
            if not animals_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Animal Species", len(animals_df))
                    if 'family' in animals_df.columns:
                        st.metric("Families Represented", animals_df['family'].nunique())
                
                with col2:
                    if 'class' in animals_df.columns:
                        st.metric("Animal Classes", animals_df['class'].nunique())
                    if 'phylum' in animals_df.columns:
                        st.metric("Phyla Represented", animals_df['phylum'].nunique())
                
                # Taxonomic breakdown
                if 'class' in animals_df.columns:
                    st.subheader("Taxonomic Distribution")
                    
                    class_dist = animals_df['class'].value_counts().head(15)
                    fig = px.bar(
                        x=class_dist.values, y=class_dist.index,
                        orientation='h',
                        title="Animal Species Count by Class",
                        labels={'x': 'Number of Species', 'y': 'Animal Class'},
                        color_discrete_sequence=[theme_colors["animal"]]
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interactive filters
                if 'phylum' in animals_df.columns and 'class' in animals_df.columns:
                    st.subheader("Interactive Filters")
                    
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        selected_phylum = st.selectbox(
                            "Filter by Phylum:", 
                            ["All"] + list(animals_df['phylum'].unique())
                        )
                    
                    with filter_col2:
                        if selected_phylum != "All":
                            filtered_animals = animals_df[animals_df['phylum'] == selected_phylum]
                            selected_class = st.selectbox(
                                "Filter by Class:",
                                ["All"] + list(filtered_animals['class'].unique())
                            )
                        else:
                            selected_class = st.selectbox(
                                "Filter by Class:",
                                ["All"] + list(animals_df['class'].unique())
                            )
                    
                    # Apply filters and show results
                    filtered_df = animals_df
                    if selected_phylum != "All":
                        filtered_df = filtered_df[filtered_df['phylum'] == selected_phylum]
                    if selected_class != "All":
                        filtered_df = filtered_df[filtered_df['class'] == selected_class]
                    
                    if len(filtered_df) < len(animals_df):
                        st.write(f"Filtered results: {len(filtered_df)} species")
                        st.dataframe(filtered_df.head(20), use_container_width=True)
            else:
                st.warning("No animal data available")
        
        with dataset_tabs[1]:
            st.subheader("Plant Dataset Explorer")
            if not plants_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Plant Species", len(plants_df))
                    if 'family' in plants_df.columns:
                        st.metric("Plant Families", plants_df['family'].nunique())
                
                with col2:
                    if 'habit' in plants_df.columns:
                        st.metric("Growth Habits", plants_df['habit'].nunique())
                    if 'duration' in plants_df.columns:
                        st.metric("Life Cycles", plants_df['duration'].nunique())
                
                # Plant characteristics visualization
                chart_cols = st.columns(2)
                
                with chart_cols[0]:
                    if 'habit' in plants_df.columns:
                        habit_dist = plants_df['habit'].value_counts()
                        fig = px.pie(
                            names=habit_dist.index, values=habit_dist.values,
                            title="Distribution by Growth Habit",
                            color_discrete_sequence=[theme_colors["plant"], theme_colors["success"], "#90EE90", "#228B22"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with chart_cols[1]:
                    if 'light' in plants_df.columns:
                        light_dist = plants_df['light'].value_counts()
                        fig = px.bar(
                            x=light_dist.index, y=light_dist.values,
                            title="Light Requirements Distribution",
                            labels={'x': 'Light Requirement', 'y': 'Number of Species'},
                            color_discrete_sequence=[theme_colors["warning"]]
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Plant search and filtering
                st.subheader("Plant Search & Filter")
                search_col1, search_col2 = st.columns(2)
                
                with search_col1:
                    search_plant = st.text_input("Search plants by name:", "")
                
                with search_col2:
                    if 'habit' in plants_df.columns:
                        filter_habit = st.selectbox(
                            "Filter by Growth Habit:",
                            ["All"] + list(plants_df['habit'].unique())
                        )
                
                # Apply search and filters
                display_plants = plants_df
                if search_plant:
                    mask = plants_df.astype(str).apply(
                        lambda x: x.str.contains(search_plant, case=False, na=False)
                    ).any(axis=1)
                    display_plants = display_plants[mask]
                
                if 'filter_habit' in locals() and filter_habit != "All":
                    display_plants = display_plants[display_plants['habit'] == filter_habit]
                
                if len(display_plants) > 0:
                    st.write(f"Found {len(display_plants)} matching plants")
                    st.dataframe(display_plants, use_container_width=True)
                else:
                    st.warning("No plants match the search criteria")
            else:
                st.warning("No plant data available")
        
        with dataset_tabs[2]:
            st.subheader("Biodiversity Dataset Explorer")
            if not biodiversity_df.empty:
                st.metric("Total Biodiversity Records", len(biodiversity_df))
                
                # Show column information
                st.subheader("Dataset Structure")
                col_info = pd.DataFrame({
                    'Column': biodiversity_df.columns,
                    'Type': [str(dtype) for dtype in biodiversity_df.dtypes],
                    'Non-Null Count': biodiversity_df.count(),
                    'Null Count': biodiversity_df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
                
                # Numeric columns analysis
                numeric_cols = biodiversity_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.subheader("Numeric Data Summary")
                    selected_numeric = st.multiselect(
                        "Select numeric columns to analyze:",
                        numeric_cols,
                        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                    )
                    
                    if selected_numeric:
                        summary_stats = biodiversity_df[selected_numeric].describe()
                        st.dataframe(summary_stats, use_container_width=True)
                        
                        # Correlation matrix
                        if len(selected_numeric) > 1:
                            st.subheader("Correlation Analysis")
                            corr_matrix = biodiversity_df[selected_numeric].corr()
                            fig = px.imshow(
                                corr_matrix,
                                title="Correlation Matrix of Biodiversity Metrics",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Sample data display
                st.subheader("Sample Data")
                sample_size = st.slider("Number of rows to display:", 5, 50, 20)
                st.dataframe(biodiversity_df.head(sample_size), use_container_width=True)
            else:
                st.warning("No biodiversity data available")
    
    with main_tabs[2]:
        st.header("🌿 Comparative Biological Analysis")
        
        comparison_tabs = st.tabs(["📊 Dataset Overview", "🔍 Cross-Dataset Analysis", "📈 Diversity Metrics"])
        
        with comparison_tabs[0]:
            st.subheader("Dataset Comparison Overview")
            
            # Create comparison metrics
            comparison_data = []
            
            if not animals_df.empty:
                animal_metrics = {
                    'Dataset': 'Animals',
                    'Total Records': len(animals_df),
                    'Unique Families': animals_df['family'].nunique() if 'family' in animals_df.columns else 0,
                    'Unique Genera': animals_df['genus'].nunique() if 'genus' in animals_df.columns else 0,
                    'Data Completeness': f"{(1 - animals_df.isnull().sum().sum() / (len(animals_df) * len(animals_df.columns))) * 100:.1f}%"
                }
                comparison_data.append(animal_metrics)
            
            if not plants_df.empty:
                plant_metrics = {
                    'Dataset': 'Plants',
                    'Total Records': len(plants_df),
                    'Unique Families': plants_df['family'].nunique() if 'family' in plants_df.columns else 0,
                    'Unique Genera': 0,  # Plants dataset doesn't have genus column
                    'Data Completeness': f"{(1 - plants_df.isnull().sum().sum() / (len(plants_df) * len(plants_df.columns))) * 100:.1f}%"
                }
                comparison_data.append(plant_metrics)
            
            if not biodiversity_df.empty:
                biodiversity_metrics = {
                    'Dataset': 'Biodiversity',
                    'Total Records': len(biodiversity_df),
                    'Unique Families': 0,  # Structure varies
                    'Unique Genera': 0,
                    'Data Completeness': f"{(1 - biodiversity_df.isnull().sum().sum() / (len(biodiversity_df) * len(biodiversity_df.columns))) * 100:.1f}%"
                }
                comparison_data.append(biodiversity_metrics)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display metrics
                metric_cols = st.columns(len(comparison_data))
                for i, (_, row) in enumerate(comparison_df.iterrows()):
                    with metric_cols[i]:
                        st.metric(f"{row['Dataset']} Records", f"{row['Total Records']:,}")
                        st.metric("Families", f"{row['Unique Families']:,}")
                        st.metric("Completeness", row['Data Completeness'])
                
                # Comparison chart
                fig = px.bar(
                    comparison_df, x='Dataset', y='Total Records',
                    title="Records Count by Dataset",
                    color='Dataset',
                    color_discrete_sequence=[theme_colors["animal"], theme_colors["plant"], theme_colors["biodiversity"]]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with comparison_tabs[1]:
            st.subheader("Cross-Dataset Analysis")
            
            if not animals_df.empty and not plants_df.empty:
                st.write("**Family Distribution Comparison**")
                
                # Compare family distributions
                animal_families = animals_df['family'].value_counts().head(10) if 'family' in animals_df.columns else pd.Series()
                plant_families = plants_df['family'].value_counts().head(10) if 'family' in plants_df.columns else pd.Series()
                
                if not animal_families.empty and not plant_families.empty:
                    comparison_chart_cols = st.columns(2)
                    
                    with comparison_chart_cols[0]:
                        fig_animals = px.bar(
                            x=animal_families.values, y=animal_families.index,
                            orientation='h',
                            title="Top Animal Families",
                            labels={'x': 'Species Count', 'y': 'Family'},
                            color_discrete_sequence=[theme_colors["animal"]]
                        )
                        st.plotly_chart(fig_animals, use_container_width=True)
                    
                    with comparison_chart_cols[1]:
                        fig_plants = px.bar(
                            x=plant_families.values, y=plant_families.index,
                            orientation='h',
                            title="Top Plant Families",
                            labels={'x': 'Species Count', 'y': 'Family'},
                            color_discrete_sequence=[theme_colors["plant"]]
                        )
                        st.plotly_chart(fig_plants, use_container_width=True)
                
                # Common families analysis
                if 'family' in animals_df.columns and 'family' in plants_df.columns:
                    animal_fam_set = set(animals_df['family'].dropna())
                    plant_fam_set = set(plants_df['family'].dropna())
                    common_families = animal_fam_set.intersection(plant_fam_set)
                    
                    if common_families:
                        st.write(f"**Families present in both datasets:** {len(common_families)}")
                        st.write(", ".join(sorted(common_families)))
                    else:
                        st.info("No common families found between animal and plant datasets")
            else:
                st.info("Both animal and plant datasets are needed for cross-dataset analysis")
        
        with comparison_tabs[2]:
            st.subheader("Diversity Metrics & Analysis")
            
            # Calculate diversity indices
            diversity_metrics = []
            
            if not animals_df.empty and 'family' in animals_df.columns:
                animal_diversity = {
                    'Dataset': 'Animals',
                    'Total Species': len(animals_df),
                    'Family Richness': animals_df['family'].nunique(),
                    'Genus Richness': animals_df['genus'].nunique() if 'genus' in animals_df.columns else 0,
                    'Simpson Index (approx)': 1 - ((animals_df['family'].value_counts() / len(animals_df)) ** 2).sum()
                }
                diversity_metrics.append(animal_diversity)
            
            if not plants_df.empty and 'family' in plants_df.columns:
                plant_diversity = {
                    'Dataset': 'Plants', 
                    'Total Species': len(plants_df),
                    'Family Richness': plants_df['family'].nunique(),
                    'Genus Richness': 0,
                    'Simpson Index (approx)': 1 - ((plants_df['family'].value_counts() / len(plants_df)) ** 2).sum()
                }
                diversity_metrics.append(plant_diversity)
            
            if diversity_metrics:
                diversity_df = pd.DataFrame(diversity_metrics)
                
                # Display diversity metrics
                st.dataframe(diversity_df, use_container_width=True)
                
                # Diversity comparison chart
                fig = px.bar(
                    diversity_df, x='Dataset', y='Family Richness',
                    title="Family Richness Comparison",
                    color='Dataset',
                    color_discrete_sequence=[theme_colors["animal"], theme_colors["plant"]]
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Simpson's Diversity Index comparison
                fig2 = px.bar(
                    diversity_df, x='Dataset', y='Simpson Index (approx)',
                    title="Simpson's Diversity Index Comparison",
                    color='Dataset',
                    color_discrete_sequence=[theme_colors["animal"], theme_colors["plant"]]
                )
                fig2.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig2, use_container_width=True)
                
                st.info("""
                **Diversity Index Interpretation:**
                - Simpson's Index ranges from 0 to 1
                - Higher values indicate greater diversity
                - Values closer to 1 suggest more evenly distributed families
                """)
            else:
                st.warning("Insufficient data for diversity calculations")

def main():
    """Main application entry point"""
    build_biology_dashboard()

if __name__ == "__main__":
    main()