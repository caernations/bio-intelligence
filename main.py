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
import re
from collections import defaultdict

st.set_page_config(page_title="Bio Intelligence Dashboard", layout="wide", page_icon="🧬")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

theme_colors = {
    "primary": "#2E8B57",      # Sea Green
    "secondary": "#DAA520",    # Goldenrod
    "success": "#228B22",      # Forest Green
    "warning": "#FF8C00",      # Dark Orange
    "danger": "#DC143C",       # Crimson
    "neutral": "#708090",      # Slate Gray
    "plant": "#228B22",        # Forest Green
    "animal": "#4169E1",       # Royal Blue
    "biodiversity": "#9932CC", # Dark Orchid
    "taxonomy": "#8B0000",     # Dark Red
    "ecology": "#006400",      # Dark Green
    "morphology": "#4B0082",   # Indigo
    "temporal": "#FF6347",     # Tomato
    "conservation": "#B8860B", # Dark Goldenrod
    "molecular": "#8A2BE2"     # Blue Violet
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

def extract_query_entities(query):
    """Extract specific biological entities, numbers, and modifiers from query"""
    entities = {
        'taxa': [],
        'characteristics': [],
        'numbers': [],
        'comparatives': [],
        'locations': [],
        'colors': [],
        'sizes': [],
        'habitats': [],
        'temporal': [],
        'conditions': []
    }
    
    # Taxonomic entities
    taxa_patterns = [
        r'\b(mammal|mammals|bird|birds|fish|fishes|reptile|reptiles|amphibian|amphibians|insect|insects)\b',
        r'\b(chordata|arthropoda|vertebrat[ae]|invertebrat[ae])\b',
        r'\b(tree|trees|shrub|shrubs|herb|herbs|grass|grasses|fern|ferns)\b',
        r'\b(annual|perennial|deciduous|evergreen)\b'
    ]
    
    # Extract characteristics
    char_patterns = [
        r'\b(toxic|poisonous|edible|medicinal|carnivorous|herbivorous|omnivorous)\b',
        r'\b(endangered|threatened|vulnerable|extinct|rare|common|abundant)\b',
        r'\b(native|endemic|invasive|introduced|alien)\b',
        r'\b(flying|swimming|climbing|burrowing|terrestrial|aquatic|arboreal)\b'
    ]
    
    # Colors
    color_patterns = r'\b(red|blue|green|yellow|orange|purple|pink|white|black|brown|gray|grey)\b'
    
    # Sizes
    size_patterns = r'\b(large|small|tiny|huge|giant|massive|miniature|big|little)\b'
    
    # Habitats
    habitat_patterns = r'\b(forest|desert|ocean|lake|river|mountain|wetland|grassland|savanna|tundra|arctic|tropical|temperate)\b'
    
    # Temporal
    temporal_patterns = r'\b(spring|summer|fall|autumn|winter|breeding|migration|hibernation|bloom|flowering)\b'
    
    # Extract numbers
    number_patterns = r'\b(\d+)\b'
    
    # Comparative terms
    comparative_patterns = r'\b(most|least|largest|smallest|fastest|slowest|longest|shortest|highest|lowest)\b'
    
    query_lower = query.lower()
    
    # Extract all patterns
    for pattern in taxa_patterns:
        entities['taxa'].extend(re.findall(pattern, query_lower, re.IGNORECASE))
    
    for pattern in char_patterns:
        entities['characteristics'].extend(re.findall(pattern, query_lower, re.IGNORECASE))
    
    entities['colors'].extend(re.findall(color_patterns, query_lower, re.IGNORECASE))
    entities['sizes'].extend(re.findall(size_patterns, query_lower, re.IGNORECASE))
    entities['habitats'].extend(re.findall(habitat_patterns, query_lower, re.IGNORECASE))
    entities['temporal'].extend(re.findall(temporal_patterns, query_lower, re.IGNORECASE))
    entities['numbers'].extend(re.findall(number_patterns, query_lower))
    entities['comparatives'].extend(re.findall(comparative_patterns, query_lower, re.IGNORECASE))
    
    return entities

def interpret_bio_query_advanced(query):
    """Enhanced interpretation of natural language queries about biological data"""
    q = query.lower()
    entities = extract_query_entities(query)
    
    # Determine query complexity and type
    query_complexity = "simple"
    if len(entities['comparatives']) > 0 or len(entities['numbers']) > 0:
        query_complexity = "analytical"
    if any(word in q for word in ["compare", "relationship", "correlation", "trend"]):
        query_complexity = "comparative"
    if any(word in q for word in ["predict", "model", "analysis", "pattern"]):
        query_complexity = "advanced"
    
    # Enhanced categorization with subcategories
    categories = {
        "taxonomic_classification": {
            "keywords": ["kingdom", "phylum", "class", "order", "family", "genus", "species", 
                        "taxonomy", "taxonomic", "classification", "linnean", "scientific name",
                        "binomial", "nomenclature", "phylogeny", "evolutionary"],
            "subcategories": ["hierarchy", "classification", "nomenclature", "phylogeny"],
            "priority": 10
        },
        
        "animal_diversity": {
            "keywords": ["animal", "animals", "mammal", "mammals", "bird", "birds", "fish", "fishes",
                        "reptile", "reptiles", "amphibian", "amphibians", "insect", "insects",
                        "vertebrate", "vertebrates", "invertebrate", "invertebrates", "chordata",
                        "arthropoda", "mollusk", "cnidarian"],
            "subcategories": ["vertebrates", "invertebrates", "marine", "terrestrial", "aerial"],
            "priority": 9
        },
        
        "plant_ecology": {
            "keywords": ["plant", "plants", "flora", "flower", "flowers", "bloom", "blooming", 
                        "leaf", "leaves", "tree", "trees", "shrub", "shrubs", "herb", "herbs",
                        "grass", "grasses", "fern", "ferns", "moss", "algae", "photosynthesis"],
            "subcategories": ["flowering", "non-flowering", "woody", "herbaceous", "aquatic"],
            "priority": 9
        },
        
        "ecological_interactions": {
            "keywords": ["ecology", "ecosystem", "habitat", "environment", "adaptation", "niche",
                        "predator", "prey", "symbiosis", "mutualism", "parasitism", "competition",
                        "food web", "food chain", "trophic", "pollination", "dispersal"],
            "subcategories": ["predation", "symbiosis", "competition", "pollination", "food_webs"],
            "priority": 8
        },
        
        "morphological_analysis": {
            "keywords": ["size", "length", "height", "weight", "color", "colour", "shape", "structure",
                        "anatomy", "morphology", "physical", "characteristics", "features", "body",
                        "wing", "tail", "beak", "root", "stem", "petal"],
            "subcategories": ["size_analysis", "color_patterns", "structural_features", "anatomy"],
            "priority": 7
        },
        
        "physiological_processes": {
            "keywords": ["physiology", "metabolism", "respiration", "circulation", "digestion",
                        "reproduction", "growth", "development", "homeostasis", "enzyme", "hormone"],
            "subcategories": ["metabolism", "reproduction", "development", "regulation"],
            "priority": 7
        },
        
        "behavioral_ecology": {
            "keywords": ["behavior", "behaviour", "migration", "mating", "breeding", "nesting",
                        "feeding", "foraging", "social", "territorial", "communication", "courtship"],
            "subcategories": ["migration", "mating", "feeding", "social_behavior", "communication"],
            "priority": 6
        },
        
        "conservation_biology": {
            "keywords": ["conservation", "endangered", "threatened", "vulnerable", "extinct", "extinction",
                        "protected", "rare", "iucn", "red list", "biodiversity loss", "habitat loss"],
            "subcategories": ["threat_status", "conservation_efforts", "habitat_protection", "species_recovery"],
            "priority": 8
        },
        
        "geographic_distribution": {
            "keywords": ["geographic", "geographical", "distribution", "range", "location", "region",
                        "continent", "country", "endemic", "native", "invasive", "introduced",
                        "biogeography", "zoogeography", "phytogeography"],
            "subcategories": ["native_range", "invasive_species", "endemic_species", "global_distribution"],
            "priority": 6
        },
        
        "temporal_patterns": {
            "keywords": ["season", "seasonal", "spring", "summer", "fall", "autumn", "winter",
                        "bloom period", "flowering time", "migration timing", "lifecycle", "life cycle",
                        "breeding season", "hibernation", "dormancy"],
            "subcategories": ["seasonal_cycles", "life_cycles", "phenology", "breeding_patterns"],
            "priority": 6
        },
        
        "environmental_factors": {
            "keywords": ["climate", "temperature", "precipitation", "humidity", "light", "soil",
                        "moisture", "ph", "salinity", "altitude", "depth", "pressure"],
            "subcategories": ["climate_factors", "soil_conditions", "water_quality", "atmospheric_conditions"],
            "priority": 7
        },
        
        "comparative_analysis": {
            "keywords": ["compare", "comparison", "versus", "vs", "difference", "differences",
                        "similar", "similarity", "related", "relationship", "correlation"],
            "subcategories": ["species_comparison", "trait_comparison", "statistical_comparison", "evolutionary_comparison"],
            "priority": 9
        },
        
        "biodiversity_metrics": {
            "keywords": ["biodiversity", "diversity", "richness", "abundance", "evenness",
                        "shannon", "simpson", "species richness", "alpha diversity", "beta diversity",
                        "gamma diversity", "community", "assemblage"],
            "subcategories": ["alpha_diversity", "beta_diversity", "community_structure", "diversity_indices"],
            "priority": 8
        },
        
        "evolutionary_biology": {
            "keywords": ["evolution", "evolutionary", "adaptation", "natural selection", "genetic",
                        "genetics", "phylogeny", "phylogenetic", "ancestor", "descent", "speciation"],
            "subcategories": ["natural_selection", "phylogenetics", "adaptation", "speciation"],
            "priority": 7
        }
    }
    
    # Score each category
    category_scores = {}
    for category, info in categories.items():
        score = 0
        words_found = []
        for keyword in info["keywords"]:
            if keyword in q:
                score += info["priority"]
                words_found.append(keyword)
        
        # Bonus for entity matches
        if category == "animal_diversity" and entities['taxa']:
            score += 5
        if category == "plant_ecology" and any(t in entities['taxa'] for t in ['tree', 'shrub', 'herb']):
            score += 5
        if category == "morphological_analysis" and (entities['colors'] or entities['sizes']):
            score += 5
        if category == "conservation_biology" and any(c in entities['characteristics'] for c in ['endangered', 'threatened', 'rare']):
            score += 5
        
        if score > 0:
            category_scores[category] = {
                'score': score,
                'words_found': words_found,
                'entities': entities
            }
    
    # Return the highest scoring category or default
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1]['score'])
        return {
            'category': best_category[0],
            'details': best_category[1],
            'complexity': query_complexity,
            'all_scores': category_scores
        }
    else:
        return {
            'category': 'general_biology',
            'details': {'score': 1, 'words_found': [], 'entities': entities},
            'complexity': query_complexity,
            'all_scores': {}
        }

def filter_biological_data_advanced(query_result, query=None, animals_df=None, plants_df=None, biodiversity_df=None):
    """Filtering and analysis of biological data based on enhanced query interpretation"""
    
    category = query_result['category']
    entities = query_result['details']['entities']
    complexity = query_result['complexity']
    
    # Initialize return values
    result_df = pd.DataFrame()
    x_col = None
    y_col = None
    title = "Biological Analysis"
    additional_data = {}
    visualization_type = "bar"
    
    if category == "taxonomic_classification":
        if animals_df.empty:
            return pd.DataFrame(), None, None, "No animal data available", {}, "bar"
        
        # Advanced taxonomic analysis
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
        available_levels = [level for level in taxonomy_levels if level in animals_df.columns]
        
        # Create hierarchical analysis
        hierarchical_data = []
        for level in available_levels:
            counts = animals_df[level].value_counts()
            for category_name, count in counts.items():
                hierarchical_data.append({
                    'taxonomic_level': level.title(),
                    'category': category_name,
                    'count': count,
                    'level_order': taxonomy_levels.index(level)
                })
        
        result_df = pd.DataFrame(hierarchical_data)
        
        # Create additional analyses
        additional_data = {
            'class_distribution': animals_df['class'].value_counts().head(15).reset_index(),
            'family_diversity': animals_df.groupby('class')['family'].nunique().reset_index().rename(columns={'family': 'family_count'}),
            'taxonomic_tree': create_taxonomic_tree(animals_df)
        }
        
        x_col = "taxonomic_level"
        y_col = "count"
        title = "Taxonomic Classification Analysis"
        visualization_type = "sunburst"
        
    elif category == "animal_diversity":
        if animals_df.empty:
            return pd.DataFrame(), None, None, "No animal data available", {}, "bar"
        
        # Focus on animal groups analysis
        if 'class' in animals_df.columns:
            class_analysis = animals_df['class'].value_counts().reset_index()
            class_analysis.columns = ['class', 'species_count']
            
            # Add vertebrate/invertebrate classification
            vertebrate_classes = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia', 'Actinopterygii', 'Chondrichthyes']
            class_analysis['group'] = class_analysis['class'].apply(
                lambda x: 'Vertebrates' if any(vc in str(x) for vc in vertebrate_classes) else 'Invertebrates'
            )
            
            result_df = class_analysis
            
            # Additional analyses
            additional_data = {
                'vertebrate_distribution': class_analysis[class_analysis['group'] == 'Vertebrates'],
                'invertebrate_distribution': class_analysis[class_analysis['group'] == 'Invertebrates'],
                'phylum_analysis': animals_df['phylum'].value_counts().head(10).reset_index()
            }
            
            x_col = "class"
            y_col = "species_count"
            title = "Animal Diversity Analysis"
            visualization_type = "treemap"
    
    elif category == "plant_ecology":
        if plants_df.empty:
            return pd.DataFrame(), None, None, "No plant data available", {}, "bar"
        
        # Plant ecological analysis
        ecological_data = []
        
        # Analyze multiple ecological dimensions
        eco_dimensions = {
            'Habitat': 'habit',
            'Light Requirements': 'light',
            'Soil Moisture': 'soil_moisture',
            'Duration': 'duration',
            'Family': 'family'
        }
        
        for dim_name, column in eco_dimensions.items():
            if column in plants_df.columns:
                counts = plants_df[column].value_counts()
                for category_name, count in counts.items():
                    ecological_data.append({
                        'dimension': dim_name,
                        'category': str(category_name),
                        'count': count
                    })
        
        result_df = pd.DataFrame(ecological_data)
        
        # Additional detailed analyses
        additional_data = {
            'habitat_light_cross': create_cross_analysis(plants_df, 'habit', 'light'),
            'bloom_analysis': analyze_bloom_periods(plants_df),
            'height_distribution': analyze_plant_heights(plants_df),
            'family_ecology': analyze_family_ecology(plants_df)
        }
        
        x_col = "dimension"
        y_col = "count"
        title = "Plant Ecological Characteristics"
        visualization_type = "bar"
    
    elif category == "morphological_analysis":
        # Handle morphological queries for both plants and animals
        morph_data = []
        
        if not animals_df.empty and entities['colors']:
            # Color analysis in common names
            for color in entities['colors']:
                color_matches = animals_df[animals_df['commonname'].str.contains(color, case=False, na=False)]
                if not color_matches.empty:
                    morph_data.append({
                        'type': 'Animal',
                        'characteristic': f"{color.title()} coloration",
                        'count': len(color_matches),
                        'examples': color_matches['commonname'].head(3).tolist()
                    })
        
        if not plants_df.empty and entities['colors']:
            for color in entities['colors']:
                color_matches = plants_df[plants_df['color'].str.contains(color, case=False, na=False)]
                if not color_matches.empty:
                    morph_data.append({
                        'type': 'Plant',
                        'characteristic': f"{color.title()} flowers",
                        'count': len(color_matches),
                        'examples': color_matches['common_name'].head(3).tolist()
                    })
        
        if morph_data:
            result_df = pd.DataFrame(morph_data)
            x_col = "characteristic"
            y_col = "count"
            title = "Morphological Characteristics Analysis"
            visualization_type = "bar"
        else:
            # General morphological analysis
            if not plants_df.empty:
                height_analysis = analyze_plant_heights(plants_df)
                if height_analysis is not None:
                    result_df = height_analysis
                    x_col = "height_category"
                    y_col = "count"
                    title = "Plant Height Distribution"
    
    elif category == "comparative_analysis":
        # Create comparative analysis between datasets
        comparison_data = []
        
        if not animals_df.empty:
            comparison_data.append({
                'dataset': 'Animals',
                'total_species': len(animals_df),
                'unique_families': animals_df['family'].nunique() if 'family' in animals_df.columns else 0,
                'unique_orders': animals_df['order'].nunique() if 'order' in animals_df.columns else 0,
                'data_type': 'Taxonomic'
            })
        
        if not plants_df.empty:
            comparison_data.append({
                'dataset': 'Plants',
                'total_species': len(plants_df),
                'unique_families': plants_df['family'].nunique() if 'family' in plants_df.columns else 0,
                'unique_orders': 0,  # Plants don't have order in this dataset
                'data_type': 'Ecological'
            })
        
        if comparison_data:
            result_df = pd.DataFrame(comparison_data)
            x_col = "dataset"
            y_col = "total_species"
            title = "Cross-Dataset Species Comparison"
            visualization_type = "bar"
    
    # Default fallback
    if result_df.empty:
        # Create a general overview
        overview_data = []
        
        if not animals_df.empty:
            overview_data.append({
                'category': 'Animal Species',
                'count': len(animals_df),
                'details': f"{animals_df['class'].nunique()} classes" if 'class' in animals_df.columns else "Various taxa"
            })
        
        if not plants_df.empty:
            overview_data.append({
                'category': 'Plant Species',
                'count': len(plants_df),
                'details': f"{plants_df['family'].nunique()} families" if 'family' in plants_df.columns else "Various families"
            })
        
        result_df = pd.DataFrame(overview_data)
        x_col = "category"
        y_col = "count"
        title = "Biological Data Overview"
        visualization_type = "bar"
    
    return result_df, x_col, y_col, title, additional_data, visualization_type

# Helper functions for advanced analyses
def create_taxonomic_tree(animals_df):
    """Create hierarchical taxonomic tree data"""
    if 'phylum' not in animals_df.columns or 'class' not in animals_df.columns:
        return None
    
    tree_data = []
    for phylum in animals_df['phylum'].unique():
        phylum_data = animals_df[animals_df['phylum'] == phylum]
        classes = phylum_data['class'].value_counts()
        
        for class_name, count in classes.items():
            tree_data.append({
                'phylum': phylum,
                'class': class_name,
                'count': count
            })
    
    return pd.DataFrame(tree_data)

def create_cross_analysis(df, col1, col2):
    """Create cross-tabulation analysis"""
    if col1 not in df.columns or col2 not in df.columns:
        return None
    
    cross_tab = pd.crosstab(df[col1], df[col2])
    return cross_tab.reset_index()

def analyze_bloom_periods(plants_df):
    """Analyze plant bloom periods"""
    if 'bloom_period' not in plants_df.columns:
        return None
    
    bloom_data = plants_df['bloom_period'].value_counts().reset_index()
    bloom_data.columns = ['bloom_period', 'count']
    return bloom_data

def analyze_plant_heights(plants_df):
    """Analyze plant height distributions"""
    if 'height' not in plants_df.columns:
        return None
    
    # Parse height data and categorize
    height_categories = []
    for height in plants_df['height'].dropna():
        try:
            # Extract numeric values from height strings
            height_str = str(height).lower()
            if 'inch' in height_str or '"' in height_str:
                # Handle inches
                numbers = re.findall(r'\d+', height_str)
                if numbers:
                    height_val = int(numbers[0]) / 12  # Convert to feet
                else:
                    continue
            else:
                # Handle feet
                numbers = re.findall(r'\d+', height_str)
                if numbers:
                    height_val = int(numbers[0])
                else:
                    continue
            
            if height_val < 3:
                height_categories.append("Low (< 3 ft)")
            elif height_val < 10:
                height_categories.append("Medium (3-10 ft)")
            elif height_val < 30:
                height_categories.append("Tall (10-30 ft)")
            else:
                height_categories.append("Very Tall (> 30 ft)")
        except:
            continue
    
    if height_categories:
        height_counts = pd.Series(height_categories).value_counts().reset_index()
        height_counts.columns = ['height_category', 'count']
        return height_counts
    
    return None

def analyze_family_ecology(plants_df):
    """Analyze ecological patterns by plant family"""
    if 'family' not in plants_df.columns:
        return None
    
    family_ecology = []
    eco_cols = ['habit', 'light', 'soil_moisture', 'duration']
    
    for family in plants_df['family'].value_counts().head(10).index:
        family_data = plants_df[plants_df['family'] == family]
        
        eco_profile = {
            'family': family,
            'species_count': len(family_data)
        }
        
        for col in eco_cols:
            if col in plants_df.columns:
                most_common = family_data[col].mode()
                if len(most_common) > 0:
                    eco_profile[f'typical_{col}'] = most_common.iloc[0]
        
        family_ecology.append(eco_profile)
    
    return pd.DataFrame(family_ecology)

def create_advanced_visualization(df, plot_type, x_col, y_col, title=None, additional_data=None, category="general"):
    """Create advanced visualizations for biological data with multiple chart types"""
    try:
        # Enhanced color palettes based on biological category
        color_palettes = {
            "plant": ["#228B22", "#32CD32", "#90EE90", "#006400", "#ADFF2F"],
            "animal": ["#4169E1", "#87CEEB", "#4682B4", "#1E90FF", "#6495ED"],
            "taxonomy": ["#8B0000", "#DC143C", "#FF6347", "#CD5C5C", "#F08080"],
            "ecology": ["#006400", "#228B22", "#32CD32", "#90EE90", "#ADFF2F"],
            "morphology": ["#4B0082", "#8A2BE2", "#9370DB", "#BA55D3", "#DDA0DD"],
            "conservation": ["#B8860B", "#DAA520", "#FFD700", "#F0E68C", "#FFFFE0"],
            "general": [theme_colors["primary"], theme_colors["secondary"], theme_colors["biodiversity"]]
        }
        
        colors = color_palettes.get(category, color_palettes["general"])
        
        if plot_type == "sunburst":
            # Enhanced sunburst for hierarchical data
            if 'taxonomic_level' in df.columns and 'category' in df.columns:
                fig = px.sunburst(
                    df, 
                    path=['taxonomic_level', 'category'], 
                    values='count',
                    title=title or "Taxonomic Hierarchy",
                    color_discrete_sequence=colors,
                    maxdepth=3
                )
                fig.update_traces(textinfo="label+percent parent")
                fig.update_layout(font_size=12)
            else:
                return None
                
        elif plot_type == "treemap":
            # Enhanced treemap with better labeling
            if len(df.columns) >= 3:
                fig = px.treemap(
                    df, 
                    path=[x_col], 
                    values=y_col,
                    title=title or f"Distribution of {y_col.replace('_', ' ').title()}",
                    color_discrete_sequence=colors
                )
                fig.update_traces(textinfo="label+value+percent parent")
                fig.update_layout(font_size=11)
            else:
                return None
                
        elif plot_type == "sankey":
            # Sankey diagram for flow analysis
            if 'source' in df.columns and 'target' in df.columns and 'value' in df.columns:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=list(set(df['source'].tolist() + df['target'].tolist())),
                        color=colors[0]
                    ),
                    link=dict(
                        source=df['source'].tolist(),
                        target=df['target'].tolist(),
                        value=df['value'].tolist()
                    )
                )])
                fig.update_layout(title_text=title or "Flow Analysis", font_size=10)
            else:
                return None
                
        elif plot_type == "radar":
            # Radar chart for multi-dimensional analysis
            if len(df.columns) >= 4:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 3:
                    fig = go.Figure()
                    
                    for idx, row in df.head(5).iterrows():  # Limit to 5 items for readability
                        fig.add_trace(go.Scatterpolar(
                            r=[row[col] for col in numeric_cols[:6]],  # Limit to 6 dimensions
                            theta=numeric_cols[:6],
                            fill='toself',
                            name=str(row[x_col]) if x_col in row else f"Item {idx}",
                            line_color=colors[idx % len(colors)]
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max([df[col].max() for col in numeric_cols[:6]])]
                            )),
                        showlegend=True,
                        title=title or "Multi-dimensional Analysis"
                    )
                else:
                    return None
            else:
                return None
                
        elif plot_type == "heatmap":
            # Enhanced heatmap
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(
                        corr_matrix,
                        title=title or "Correlation Heatmap",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        aspect="auto"
                    )
                    fig.update_xaxes(side="bottom")
                else:
                    return None
            else:
                return None
                
        elif plot_type == "network":
            # Network diagram for relationships
            if 'source' in df.columns and 'target' in df.columns:
                # This would require networkx, simplified version
                fig = px.scatter(
                    df, x='source', y='target', 
                    title=title or "Network Relationships",
                    color_discrete_sequence=colors
                )
            else:
                return None
                
        elif plot_type == "violin":
            # Violin plot for distribution analysis
            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                fig = px.violin(
                    df, x=x_col, y=y_col,
                    title=title or f"Distribution of {y_col.replace('_', ' ').title()}",
                    color_discrete_sequence=colors,
                    box=True
                )
            else:
                return None
                
        elif plot_type == "parallel_coordinates":
            # Parallel coordinates for multi-dimensional data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                fig = px.parallel_coordinates(
                    df, 
                    dimensions=numeric_cols[:6],  # Limit dimensions
                    title=title or "Parallel Coordinates Analysis",
                    color_continuous_scale="Viridis"
                )
            else:
                return None
                
        elif plot_type == "bar":
            # Enhanced bar chart
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()},
                color_discrete_sequence=colors,
                text=y_col
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=50, b=100),
                height=500
            )
            
        elif plot_type == "scatter":
            # Enhanced scatter plot
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                size_col = numeric_cols[2] if len(numeric_cols) > 2 else None
                fig = px.scatter(
                    df, 
                    x=numeric_cols[0], 
                    y=numeric_cols[1],
                    size=size_col,
                    title=title or f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                    color_discrete_sequence=colors,
                    hover_data=df.columns.tolist()[:5]
                )
                
                # Add trendline if enough data points
                if len(df) > 5:
                    fig.add_traces(px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                            trendline="ols").data[1:])
            else:
                return None
                
        elif plot_type == "pie":
            # Enhanced pie chart
            fig = px.pie(
                df, names=x_col, values=y_col,
                title=title or f"Distribution of {y_col.replace('_', ' ').title()}",
                color_discrete_sequence=colors
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
        else:
            # Default to enhanced bar chart
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title or f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}",
                color_discrete_sequence=colors
            )
        
        # Apply consistent styling
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None

def generate_advanced_insights(category, df, x_col, y_col, additional_data, entities):
    """Generate sophisticated biological insights and additional visualizations"""
    charts = []
    insights = []
    
    if category == "taxonomic_classification":
        # Phylogenetic diversity analysis
        if additional_data and 'class_distribution' in additional_data:
            class_data = additional_data['class_distribution']
            
            # Vertebrate vs Invertebrate analysis
            vertebrate_classes = ['Mammalia', 'Aves', 'Reptilia', 'Amphibia', 'Actinopterygii', 'Chondrichthyes']
            class_data['group'] = class_data['class'].apply(
                lambda x: 'Vertebrates' if any(vc in str(x) for vc in vertebrate_classes) else 'Invertebrates'
            )
            
            group_summary = class_data.groupby('group')['count'].sum().reset_index()
            
            group_fig = px.pie(
                group_summary, names='group', values='count',
                title="Vertebrates vs Invertebrates Distribution",
                color_discrete_sequence=[theme_colors["animal"], theme_colors["secondary"]]
            )
            charts.append(("Vertebrate Classification", group_fig))
            
            insights.append(f"**Taxonomic Diversity**: Found {len(class_data)} different animal classes")
            insights.append(f"**Vertebrate Ratio**: {group_summary[group_summary['group']=='Vertebrates']['count'].sum()/(group_summary['count'].sum())*100:.1f}% vertebrates")
        
        # Family diversity by class
        if additional_data and 'family_diversity' in additional_data:
            family_div = additional_data['family_diversity']
            if not family_div.empty:
                family_fig = px.bar(
                    family_div, x='class', y='family_count',
                    title="Family Diversity by Animal Class",
                    color_discrete_sequence=[theme_colors["taxonomy"]]
                )
                family_fig.update_layout(xaxis_tickangle=-45)
                charts.append(("Family Diversity", family_fig))
                
                max_diversity_class = family_div.loc[family_div['family_count'].idxmax(), 'class']
                insights.append(f"**Highest Family Diversity**: {max_diversity_class} with {family_div['family_count'].max()} families")
    
    elif category == "animal_diversity":
        if additional_data:
            # Vertebrate analysis
            if 'vertebrate_distribution' in additional_data:
                vert_data = additional_data['vertebrate_distribution']
                if not vert_data.empty:
                    vert_fig = px.bar(
                        vert_data, x='class', y='species_count',
                        title="Vertebrate Class Distribution",
                        color_discrete_sequence=[theme_colors["animal"]]
                    )
                    charts.append(("Vertebrate Analysis", vert_fig))
                    
                    insights.append(f"**Vertebrate Classes**: {len(vert_data)} vertebrate classes represented")
                    insights.append(f"**Most Diverse Vertebrate Class**: {vert_data.iloc[0]['class']} ({vert_data.iloc[0]['species_count']} species)")
            
            # Invertebrate analysis
            if 'invertebrate_distribution' in additional_data:
                invert_data = additional_data['invertebrate_distribution']
                if not invert_data.empty:
                    invert_fig = px.bar(
                        invert_data, x='class', y='species_count',
                        title="Invertebrate Class Distribution",
                        color_discrete_sequence=[theme_colors["biodiversity"]]
                    )
                    invert_fig.update_layout(xaxis_tickangle=-45)
                    charts.append(("Invertebrate Analysis", invert_fig))
                    
                    insights.append(f"**Invertebrate Diversity**: {invert_data['species_count'].sum()} invertebrate species")
            
            # Phylum analysis
            if 'phylum_analysis' in additional_data:
                phylum_data = additional_data['phylum_analysis']
                phylum_fig = px.treemap(
                    phylum_data, path=['phylum'], values='count',
                    title="Species Distribution by Phylum",
                    color_discrete_sequence=[theme_colors["taxonomy"], theme_colors["animal"]]
                )
                charts.append(("Phylum Distribution", phylum_fig))
    
    elif category == "plant_ecology":
        if additional_data:
            # Habitat-Light cross analysis
            if 'habitat_light_cross' in additional_data:
                cross_data = additional_data['habitat_light_cross']
                if cross_data is not None and not cross_data.empty:
                    # Create heatmap of habitat vs light requirements
                    heatmap_fig = px.imshow(
                        cross_data.set_index(cross_data.columns[0]).values,
                        labels=dict(x="Light Requirement", y="Growth Habit", color="Species Count"),
                        x=cross_data.columns[1:].tolist(),
                        y=cross_data.iloc[:, 0].tolist(),
                        title="Plant Habitat vs Light Requirements",
                        color_continuous_scale="Greens"
                    )
                    charts.append(("Ecological Cross-Analysis", heatmap_fig))
            
            # Bloom period analysis
            if 'bloom_analysis' in additional_data:
                bloom_data = additional_data['bloom_analysis']
                if bloom_data is not None and not bloom_data.empty:
                    bloom_fig = px.bar(
                        bloom_data, x='bloom_period', y='count',
                        title="Plant Blooming Periods",
                        color_discrete_sequence=[theme_colors["plant"]]
                    )
                    bloom_fig.update_layout(xaxis_tickangle=-45)
                    charts.append(("Blooming Patterns", bloom_fig))
                    
                    peak_bloom = bloom_data.loc[bloom_data['count'].idxmax(), 'bloom_period']
                    insights.append(f"**Peak Blooming Period**: {peak_bloom}")
            
            # Height distribution
            if 'height_distribution' in additional_data:
                height_data = additional_data['height_distribution']
                if height_data is not None and not height_data.empty:
                    height_fig = px.pie(
                        height_data, names='height_category', values='count',
                        title="Plant Height Distribution",
                        color_discrete_sequence=[theme_colors["morphology"], theme_colors["plant"], theme_colors["success"], theme_colors["warning"]]
                    )
                    charts.append(("Height Distribution", height_fig))
                    
                    insights.append(f"**Height Diversity**: Plants range from ground covers to tall trees")
            
            # Family ecology patterns
            if 'family_ecology' in additional_data:
                family_eco = additional_data['family_ecology']
                if family_eco is not None and not family_eco.empty:
                    family_fig = px.bar(
                        family_eco, x='family', y='species_count',
                        title="Species Count by Plant Family",
                        color_discrete_sequence=[theme_colors["ecology"]]
                    )
                    family_fig.update_layout(xaxis_tickangle=-45)
                    charts.append(("Family Diversity", family_fig))
                    
                    most_diverse_family = family_eco.loc[family_eco['species_count'].idxmax(), 'family']
                    insights.append(f"**Most Diverse Family**: {most_diverse_family}")
    
    elif category == "morphological_analysis":
        # Color analysis insights
        if entities['colors']:
            insights.append(f"**Color Analysis**: Found species with {', '.join(entities['colors'])} coloration")
        
        # Size analysis insights
        if entities['sizes']:
            insights.append(f"**Size Patterns**: Analysis focused on {', '.join(entities['sizes'])} organisms")
    
    elif category == "comparative_analysis":
        # Cross-dataset comparison insights
        if not df.empty and 'total_species' in df.columns:
            total_species = df['total_species'].sum()
            insights.append(f"**Total Biodiversity**: {total_species:,} species across all datasets")
            
            if len(df) > 1:
                largest_dataset = df.loc[df['total_species'].idxmax(), 'dataset']
                insights.append(f"**Largest Dataset**: {largest_dataset} dataset contains the most species")
    
    # Add entity-based insights
    if entities['numbers']:
        insights.append(f"**Numerical Focus**: Analysis includes specific counts: {', '.join(entities['numbers'])}")
    
    if entities['comparatives']:
        insights.append(f"**Comparative Analysis**: Focus on {', '.join(entities['comparatives'])} characteristics")
    
    if entities['habitats']:
        insights.append(f"**Habitat Analysis**: Examining {', '.join(entities['habitats'])} ecosystems")
    
    return charts, insights

def build_enhanced_biology_dashboard():
    """Main dashboard interface with advanced query processing"""
    st.title("Bio Intelligence Dashboard")
    st.markdown("### Analysis of Biological Data with Intelligent Query Processing")
    
    # Load data
    with st.spinner("Loading biological datasets..."):
        animals_df, plants_df, biodiversity_df = load_biological_data()
    
    # Enhanced sidebar with advanced metrics
    with st.sidebar:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            if not animals_df.empty:
                st.metric("Animal Species", f"{len(animals_df):,}")
                if 'class' in animals_df.columns:
                    st.metric("Animal Classes", f"{animals_df['class'].nunique()}")
        
        with col2:
            if not plants_df.empty:
                st.metric("Plant Species", f"{len(plants_df):,}")
                if 'family' in plants_df.columns:
                    st.metric("Plant Families", f"{plants_df['family'].nunique()}")
        
        if not biodiversity_df.empty:
            st.metric("Biodiversity Records", f"{len(biodiversity_df):,}")
        
        # Data quality indicators
        st.divider()
        st.subheader("Data Quality")
        
        if not animals_df.empty:
            completeness = (1 - animals_df.isnull().sum().sum() / (len(animals_df) * len(animals_df.columns))) * 100
            st.progress(completeness/100, text=f"Animal Data: {completeness:.1f}% complete")
        
        if not plants_df.empty:
            completeness = (1 - plants_df.isnull().sum().sum() / (len(plants_df) * len(plants_df.columns))) * 100
            st.progress(completeness/100, text=f"Plant Data: {completeness:.1f}% complete")
        
        st.divider()
        st.subheader("Query Examples")
        example_queries = [
            "Show me all vertebrate animals",
            "What plants bloom in spring?",
            "Compare plant families by diversity",
            "Which animals are in the class Mammalia?",
            "Show plants that need full sun",
            "What are the largest animal groups?",
            "Which plants have red flowers?",
            "Show taxonomic hierarchy",
            "Compare vertebrates vs invertebrates"
        ]
        
        for query in example_queries[:5]:
            if st.button(query, key=f"example_{hash(query)}"):
                st.session_state.example_query = query
    
    # Main analysis interface
    main_tabs = st.tabs([
        "Query Analysis", 
        "Interactive Explorer"
    ])
    
    with main_tabs[0]:
        st.header("Natural Language Query Analysis")
        
        # Query input with enhanced features
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Check for example query
            default_query = ""
            if hasattr(st.session_state, 'example_query'):
                default_query = st.session_state.example_query
                delattr(st.session_state, 'example_query')
            
            user_query = st.text_area(
                "Enter your biological research question:",
                value=default_query,
                placeholder="e.g., 'Show me all mammals in the dataset and compare their family diversity' or 'What plants need wet soil and bloom in summer?'",
                height=100
            )
            
            st.markdown("**Query Features:**")
            st.markdown("- **Taxonomic queries**: Ask about specific kingdoms, phyla, classes, orders, families")
            st.markdown("- **Ecological queries**: Search by habitat, light, soil, moisture requirements")
            st.markdown("- **Morphological queries**: Find organisms by size, color, structure")
            st.markdown("- **Comparative analysis**: Compare different groups or characteristics")
            st.markdown("- **Temporal patterns**: Analyze seasonal or lifecycle patterns")
        
        with col2:
            st.subheader("Analysis Options")
            
            analysis_depth = st.selectbox("Analysis Depth:", [
                "Quick Overview",
                "Detailed Analysis", 
                "Comprehensive Study",
                "Research-Level Deep Dive"
            ])
            
            include_insights = st.checkbox("Generate Insights", value=True)
            include_statistics = st.checkbox("Include Statistics", value=True)
            auto_visualize = st.checkbox("Auto-select Visualizations", value=True)
            
            if not auto_visualize:
                viz_preference = st.selectbox("Preferred Visualization:", [
                    "bar", "pie", "sunburst", "treemap", "scatter", 
                    "heatmap", "violin", "radar", "network"
                ])
            else:
                viz_preference = "auto"
        
        # Process query when submitted
        if user_query:
            st.info(f"Processing Query: **{user_query}**")
            
            with st.spinner("Analyzing biological data with advanced algorithms..."):
                # Enhanced query interpretation
                query_result = interpret_bio_query_advanced(user_query)
                
                # Show query analysis details
                with st.expander("Query Analysis Details"):
                    st.write(f"**Detected Category**: {query_result['category'].replace('_', ' ').title()}")
                    st.write(f"**Query Complexity**: {query_result['complexity'].title()}")
                    st.write(f"**Confidence Score**: {query_result['details']['score']}")
                    
                    if query_result['details']['words_found']:
                        st.write(f"**Key Terms Found**: {', '.join(query_result['details']['words_found'])}")
                    
                    entities = query_result['details']['entities']
                    for entity_type, values in entities.items():
                        if values:
                            st.write(f"**{entity_type.title()}**: {', '.join(values)}")
                
                # Get filtered data and analysis
                result = filter_biological_data_advanced(
                    query_result, user_query, animals_df, plants_df, biodiversity_df
                )
                
                if len(result) == 6:
                    df, x_col, y_col, title, additional_data, suggested_viz = result
                else:
                    df, x_col, y_col, title, additional_data = result[:5]
                    suggested_viz = "bar"
                
                # Determine visualization type
                if auto_visualize:
                    viz_type = suggested_viz
                else:
                    viz_type = viz_preference
                
                # Determine biological category for coloring
                bio_category = "general"
                if "animal" in query_result['category']:
                    bio_category = "animal"
                elif "plant" in query_result['category']:
                    bio_category = "plant"
                elif "taxonomic" in query_result['category']:
                    bio_category = "taxonomy"
                elif "morphological" in query_result['category']:
                    bio_category = "morphology"
                elif "ecological" in query_result['category']:
                    bio_category = "ecology"
                
                # Create main visualization and insights
                analysis_tabs = st.tabs([
                    "Primary Analysis", 
                    "Detailed Insights", 
                    "Additional Visualizations",
                    "Data Explorer",
                    "AI Interpretation"
                ])
                
                with analysis_tabs[0]:
                    st.subheader(title)
                    
                    if not df.empty and x_col and y_col:
                        # Main metrics
                        metric_cols = st.columns(4)
                        
                        with metric_cols[0]:
                            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                                st.metric("Total Count", f"{df[y_col].sum():,}")
                        
                        with metric_cols[1]:
                            st.metric("Categories", f"{len(df)}")
                        
                        with metric_cols[2]:
                            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                                st.metric("Average", f"{df[y_col].mean():.1f}")
                        
                        with metric_cols[3]:
                            if y_col in df.columns and df[y_col].dtype in [np.float64, np.int64]:
                                st.metric("Maximum", f"{df[y_col].max():,}")
                        
                        # Main visualization
                        fig = create_advanced_visualization(
                            df, viz_type, x_col, y_col, title, additional_data, bio_category
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"Could not create {viz_type} visualization with current data. Showing data table instead.")
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
                        if df.empty:
                            st.warning("No data found matching your query. Try rephrasing or using different terms.")
                
                with analysis_tabs[1]:
                    st.subheader("Detailed Biological Insights")
                    
                    # Generate advanced insights
                    charts, insights = generate_advanced_insights(
                        query_result['category'], df, x_col, y_col, 
                        additional_data, query_result['details']['entities']
                    )
                    
                    if insights:
                        st.markdown("### Key Findings")
                        for insight in insights:
                            st.markdown(f"- {insight}")
                    
                    # Statistical summary if requested
                    if include_statistics and not df.empty:
                        st.markdown("### Statistical Summary")
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            stats_df = df[numeric_cols].describe().round(2)
                            st.dataframe(stats_df, use_container_width=True)
                        
                        # Correlation analysis if multiple numeric columns
                        if len(numeric_cols) > 1:
                            st.markdown("### Correlation Analysis")
                            corr_matrix = df[numeric_cols].corr()
                            
                            corr_fig = px.imshow(
                                corr_matrix,
                                title="Variable Correlations",
                                color_continuous_scale="RdBu_r",
                                zmin=-1, zmax=1
                            )
                            st.plotly_chart(corr_fig, use_container_width=True)
                
                with analysis_tabs[2]:
                    st.subheader("Additional Visualizations & Analysis")
                    
                    if charts:
                        for i, (chart_title, chart_fig) in enumerate(charts):
                            if i % 2 == 0:
                                chart_cols = st.columns(2)
                            
                            with chart_cols[i % 2]:
                                st.markdown(f"**{chart_title}**")
                                st.plotly_chart(chart_fig, use_container_width=True)
                    
                    else:
                        # Generate alternative visualizations based on data
                        if not df.empty and x_col and y_col:
                            st.markdown("**Alternative Visualizations:**")
                            
                            alt_viz_cols = st.columns(3)
                            alt_viz_types = ["pie", "treemap", "violin"] if viz_type != "bar" else ["scatter", "heatmap", "radar"]
                            
                            for idx, alt_viz in enumerate(alt_viz_types):
                                with alt_viz_cols[idx]:
                                    alt_fig = create_advanced_visualization(
                                        df, alt_viz, x_col, y_col, f"{title} - {alt_viz.title()} View", 
                                        additional_data, bio_category
                                    )
                                    if alt_fig:
                                        st.plotly_chart(alt_fig, use_container_width=True)
                
                with analysis_tabs[3]:
                    st.subheader("Interactive Data Explorer")
                    
                    if not df.empty:
                        # Enhanced data filtering
                        filter_cols = st.columns(3)
                        
                        with filter_cols[0]:
                            if len(df) > 20:
                                show_top_n = st.slider("Show top N results", 5, min(100, len(df)), 20)
                                df_display = df.head(show_top_n)
                            else:
                                df_display = df
                        
                        with filter_cols[1]:
                            sort_column = st.selectbox("Sort by:", df.columns.tolist())
                            ascending = st.checkbox("Ascending", value=False)
                        
                        with filter_cols[2]:
                            search_term = st.text_input("Search in data:")
                        
                        # Apply filters
                        if sort_column:
                            df_display = df_display.sort_values(sort_column, ascending=ascending)
                        
                        if search_term:
                            text_cols = df_display.select_dtypes(include=['object']).columns
                            if len(text_cols) > 0:
                                mask = df_display[text_cols].astype(str).apply(
                                    lambda x: x.str.contains(search_term, case=False, na=False)
                                ).any(axis=1)
                                df_display = df_display[mask]
                        
                        # Display filtered data
                        st.dataframe(df_display, use_container_width=True)
                        
                        # Data export options
                        export_cols = st.columns(3)
                        with export_cols[0]:
                            csv = df_display.to_csv(index=False)
                            st.download_button("Download CSV", csv, "biological_analysis.csv", "text/csv")
                        
                        with export_cols[1]:
                            json_data = df_display.to_json(orient='records', indent=2)
                            st.download_button("Download JSON", json_data, "biological_analysis.json", "application/json")
                        
                        # Data profiling
                        with st.expander("Data Profiling"):
                            profile_cols = st.columns(2)
                            
                            with profile_cols[0]:
                                st.write("**Column Statistics:**")
                                col_stats = pd.DataFrame({
                                    'Column': df_display.columns,
                                    'Type': [str(dtype) for dtype in df_display.dtypes],
                                    'Non-Null': df_display.count(),
                                    'Unique Values': [df_display[col].nunique() for col in df_display.columns],
                                    'Null %': [(df_display[col].isnull().sum() / len(df_display)) * 100 for col in df_display.columns]
                                })
                                st.dataframe(col_stats, use_container_width=True)
                            
                            with profile_cols[1]:
                                numeric_cols = df_display.select_dtypes(include=[np.number]).columns.tolist()
                                if numeric_cols:
                                    st.write("**Numeric Summary:**")
                                    st.dataframe(df_display[numeric_cols].describe(), use_container_width=True)
                    else:
                        st.warning("No data available for exploration")
                
                with analysis_tabs[4]:
                    st.subheader("AI-Powered Scientific Interpretation")
                    
                    if include_insights and client:
                        # Enhanced context preparation
                        context = f"""
                        Biological Analysis Results:
                        Query Category: {query_result['category'].replace('_', ' ').title()}
                        Complexity Level: {query_result['complexity']}
                        
                        Dataset Summary:
                        - Total records analyzed: {len(df) if not df.empty else 0}
                        - Analysis focus: {title}
                        
                        Key Findings:
                        {chr(10).join([f"- {insight}" for insight in insights]) if insights else "Standard analysis completed"}
                        
                        Data Characteristics:
                        {df.describe().to_string() if not df.empty and len(df.select_dtypes(include=[np.number]).columns) > 0 else "Categorical data analysis"}
                        
                        Query Entities Detected:
                        {chr(10).join([f"- {k}: {v}" for k, v in query_result['details']['entities'].items() if v])}
                        """
                        
                        if query_result['category'] == "taxonomic_classification":
                            context += f"""
                            
                            Taxonomic Analysis Context:
                            - Focus on hierarchical classification systems
                            - Linnean taxonomy principles
                            - Phylogenetic relationships
                            """
                        
                        elif query_result['category'] == "plant_ecology":
                            context += f"""
                            
                            Plant Ecology Context:
                            - Environmental adaptation strategies
                            - Habitat requirements and preferences
                            - Ecological niche analysis
                            """
                        
                        elif query_result['category'] == "animal_diversity":
                            context += f"""
                            
                            Animal Diversity Context:
                            - Evolutionary adaptations
                            - Ecological roles and behaviors
                            - Taxonomic diversity patterns
                            """
                        
                        prompt = f"""
                        Based on this biological data analysis, provide a comprehensive scientific interpretation that includes:
                        
                        1. **Biological Significance**: What do these findings tell us about the organisms or ecosystems studied?
                        2. **Ecological Implications**: How do these patterns relate to ecological principles and processes?
                        3. **Evolutionary Context**: What evolutionary or adaptive significance might these patterns have?
                        4. **Conservation Relevance**: Are there any conservation implications or concerns highlighted by this data?
                        5. **Research Directions**: What follow-up questions or research directions does this analysis suggest?
                        6. **Methodological Insights**: Comment on the analytical approach and data quality
                        
                        Context: {context}
                        
                        User's Original Query: "{user_query}"
                        
                        Please provide a detailed, scientifically accurate response that would be suitable for researchers, students, or conservation professionals. Use appropriate biological terminology while remaining accessible. Respond in English.
                        """
                        
                        with st.spinner("Generating comprehensive AI analysis..."):
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "You are a leading biological researcher and data scientist with expertise in ecology, evolution, taxonomy, and conservation biology. Provide comprehensive, scientifically accurate analyses that integrate multiple biological disciplines. Always respond in English with proper scientific terminology."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_tokens=1500,
                                    temperature=0.3,
                                )
                                ai_analysis = response.choices[0].message.content
                                
                                # Display AI analysis with better formatting
                                st.markdown(ai_analysis)
                                
                                # Add research recommendations
                                st.markdown("---")
                                st.markdown("### 🔬 Research Recommendations")
                                
                                rec_cols = st.columns(2)
                                with rec_cols[0]:
                                    st.markdown("**Immediate Actions:**")
                                    st.markdown("- Validate findings with additional data sources")
                                    st.markdown("- Consider temporal variation in patterns")
                                    st.markdown("- Examine potential confounding variables")
                                
                                with rec_cols[1]:
                                    st.markdown("**Future Research:**")
                                    st.markdown("- Expand analysis to related taxa/ecosystems")
                                    st.markdown("- Investigate mechanistic explanations")
                                    st.markdown("- Consider climate change implications")
                                
                            except Exception as e:
                                st.error(f"Error generating AI analysis: {str(e)}")
                                st.markdown(self_generate_insights(query_result, df, insights))
                    
                    else:
                        st.info("Enable AI Analysis in the sidebar for detailed scientific interpretation")
                        st.markdown(self_generate_insights(query_result, df, insights))
                        
    # Continue with the rest of the dashboard
    with main_tabs[1]:
        st.header("Interactive Biological Data Explorer")
        
        # Enhanced dataset exploration
        dataset_selection = st.selectbox("Select Primary Dataset:", [
            "Animals - Taxonomic Analysis",
            "Plants - Ecological Analysis", 
            "Biodiversity - Community Analysis",
            "Comparative - Cross-Dataset Analysis"
        ])
        
        if dataset_selection.startswith("Animals"):
            explore_animals_dataset(animals_df)
        elif dataset_selection.startswith("Plants"):
            explore_plants_dataset(plants_df)
        elif dataset_selection.startswith("Biodiversity"):
            explore_biodiversity_dataset(biodiversity_df)
        else:
            explore_comparative_analysis(animals_df, plants_df, biodiversity_df)
            

def self_generate_insights(query_result, df, insights):
    """Generate basic insights when AI is not available"""
    category = query_result['category']
    
    basic_analysis = f"""
    ## Basic Scientific Analysis
    
    **Analysis Type**: {category.replace('_', ' ').title()}
    
    **Key Observations**:
    """
    
    if insights:
        for insight in insights:
            basic_analysis += f"\n- {insight}"
    
    basic_analysis += f"""
    
    **Data Summary**:
    - Records analyzed: {len(df) if not df.empty else 0}
    - Query complexity: {query_result['complexity']}
    
    **Scientific Context**:
    """
    
    if category == "taxonomic_classification":
        basic_analysis += """
        - Taxonomic classification provides the foundation for understanding evolutionary relationships
        - Higher diversity at certain taxonomic levels may indicate adaptive radiation
        - Systematic organization helps identify patterns in biodiversity distribution
        """
    
    elif category == "plant_ecology":
        basic_analysis += """
        - Plant ecological characteristics reflect evolutionary adaptations to environmental conditions
        - Habitat requirements indicate species' environmental tolerances and limits
        - Blooming patterns often correlate with pollinator availability and seasonal resources
        """
    
    elif category == "animal_diversity":
        basic_analysis += """
        - Animal diversity patterns reflect ecological niches and evolutionary history
        - Vertebrate vs invertebrate ratios indicate ecosystem structure
        - Class-level diversity suggests different evolutionary strategies and ecological roles
        """
    
    basic_analysis += """
    
    **Recommendations**:
    - Consider additional environmental variables for more comprehensive analysis
    - Examine temporal patterns and seasonal variations
    - Investigate geographic distribution patterns
    - Validate findings with field observations where possible
    
    *Enable OpenAI API for more detailed AI-powered scientific interpretation.*
    """
    
    return basic_analysis



def explore_animals_dataset(animals_df):
    """Animal dataset exploration"""
    if animals_df.empty:
        st.warning("No animal data available")
        return
    
    st.subheader("Animal Taxonomic Explorer")
    
    # Interactive taxonomic filtering
    filter_cols = st.columns(4)
    
    with filter_cols[0]:
        if 'phylum' in animals_df.columns:
            selected_phylum = st.selectbox("Phylum:", ["All"] + sorted(animals_df['phylum'].dropna().unique()))
        else:
            selected_phylum = "All"
    
    with filter_cols[1]:
        filtered_df = animals_df if selected_phylum == "All" else animals_df[animals_df['phylum'] == selected_phylum]
        if 'class' in filtered_df.columns:
            selected_class = st.selectbox("Class:", ["All"] + sorted(filtered_df['class'].dropna().unique()))
        else:
            selected_class = "All"
    
    with filter_cols[2]:
        if selected_class != "All":
            filtered_df = filtered_df[filtered_df['class'] == selected_class]
        if 'order' in filtered_df.columns:
            selected_order = st.selectbox("Order:", ["All"] + sorted(filtered_df['order'].dropna().unique()))
        else:
            selected_order = "All"
    
    with filter_cols[3]:
        if selected_order != "All":
            filtered_df = filtered_df[filtered_df['order'] == selected_order]
        if 'family' in filtered_df.columns:
            selected_family = st.selectbox("Family:", ["All"] + sorted(filtered_df['family'].dropna().unique()))
        else:
            selected_family = "All"
    
    # Apply final filter
    if selected_family != "All":
        filtered_df = filtered_df[filtered_df['family'] == selected_family]
    
    # Display results
    result_cols = st.columns([2, 1])
    
    with result_cols[0]:
        st.write(f"**Filtered Results**: {len(filtered_df)} species")
        if len(filtered_df) > 0:
            st.dataframe(filtered_df, use_container_width=True)
    
    with result_cols[1]:
        if len(filtered_df) > 0:
            # Dynamic statistics based on current filter
            if 'family' in filtered_df.columns:
                st.metric("Families", filtered_df['family'].nunique())
            if 'genus' in filtered_df.columns:
                st.metric("Genera", filtered_df['genus'].nunique())
            st.metric("Species", len(filtered_df))
    
    # Visualization of current selection
    if len(filtered_df) > 0:
        viz_cols = st.columns(2)
        
        with viz_cols[0]:
            if 'family' in filtered_df.columns and filtered_df['family'].nunique() > 1:
                family_counts = filtered_df['family'].value_counts()
                fig = px.pie(names=family_counts.index, values=family_counts.values,
                           title="Family Distribution in Current Selection")
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_cols[1]:
            if 'genus' in filtered_df.columns and filtered_df['genus'].nunique() > 1:
                genus_counts = filtered_df['genus'].value_counts().head(10)
                fig = px.bar(x=genus_counts.values, y=genus_counts.index, orientation='h',
                           title="Top 10 Genera in Current Selection")
                st.plotly_chart(fig, use_container_width=True)

def explore_plants_dataset(plants_df):
    """Plant dataset exploration"""
    if plants_df.empty:
        st.warning("No plant data available")
        return
    
    st.subheader("🌱 Plant Ecological Explorer")
    
    # Multi-dimensional filtering
    filter_cols = st.columns(3)
    
    with filter_cols[0]:
        if 'habit' in plants_df.columns:
            selected_habits = st.multiselect("Growth Habit:", sorted(plants_df['habit'].unique()))
        else:
            selected_habits = []
    
    with filter_cols[1]:
        if 'light' in plants_df.columns:
            selected_light = st.multiselect("Light Requirements:", sorted(plants_df['light'].unique()))
        else:
            selected_light = []
    
    with filter_cols[2]:
        if 'soil_moisture' in plants_df.columns:
            selected_moisture = st.multiselect("Soil Moisture:", sorted(plants_df['soil_moisture'].unique()))
        else:
            selected_moisture = []
    
    # Apply filters
    filtered_df = plants_df.copy()
    
    if selected_habits:
        filtered_df = filtered_df[filtered_df['habit'].isin(selected_habits)]
    if selected_light:
        filtered_df = filtered_df[filtered_df['light'].isin(selected_light)]
    if selected_moisture:
        filtered_df = filtered_df[filtered_df['soil_moisture'].isin(selected_moisture)]
    
    # Display results with enhanced visualizations
    st.write(f"**Filtered Results**: {len(filtered_df)} plants")
    
    if len(filtered_df) > 0:
        # Create ecological profile visualization
        eco_viz_cols = st.columns(3)
        
        with eco_viz_cols[0]:
            if 'bloom_period' in filtered_df.columns:
                bloom_counts = filtered_df['bloom_period'].value_counts()
                fig = px.bar(x=bloom_counts.index, y=bloom_counts.values,
                           title="Blooming Periods", color_discrete_sequence=[theme_colors["plant"]])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with eco_viz_cols[1]:
            if 'color' in filtered_df.columns:
                color_counts = filtered_df['color'].value_counts().head(8)
                fig = px.pie(names=color_counts.index, values=color_counts.values,
                           title="Flower Colors")
                st.plotly_chart(fig, use_container_width=True)
        
        with eco_viz_cols[2]:
            if 'duration' in filtered_df.columns:
                duration_counts = filtered_df['duration'].value_counts()
                fig = px.bar(x=duration_counts.values, y=duration_counts.index, orientation='h',
                           title="Plant Duration Types", color_discrete_sequence=[theme_colors["ecology"]])
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed data table
        st.dataframe(filtered_df, use_container_width=True)

def explore_biodiversity_dataset(biodiversity_df):
    """Biodiversity dataset exploration"""
    if biodiversity_df.empty:
        st.warning("No biodiversity data available")
        return
    
    st.subheader("Biodiversity Community Analysis")
    
    # Show dataset structure
    st.write("**Dataset Structure:**")
    structure_info = pd.DataFrame({
        'Column': biodiversity_df.columns,
        'Type': [str(dtype) for dtype in biodiversity_df.dtypes],
        'Non-Null Count': biodiversity_df.count(),
        'Sample Values': [str(biodiversity_df[col].dropna().iloc[0]) if not biodiversity_df[col].dropna().empty else "No data" for col in biodiversity_df.columns]
    })
    st.dataframe(structure_info, use_container_width=True)
    
    # Basic analysis
    numeric_cols = biodiversity_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.write("**Numeric Data Summary:**")
        st.dataframe(biodiversity_df[numeric_cols].describe(), use_container_width=True)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.write("**Correlation Analysis:**")
            corr_matrix = biodiversity_df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Biodiversity Metrics Correlation",
                          color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    st.write("**Sample Records:**")
    st.dataframe(biodiversity_df.head(20), use_container_width=True)

def explore_comparative_analysis(animals_df, plants_df, biodiversity_df):
    """Comparative analysis across datasets"""
    st.subheader("Cross-Dataset Comparative Analysis")
    
    # Dataset overview
    comparison_metrics = []
    
    if not animals_df.empty:
        comparison_metrics.append({
            'Dataset': 'Animals',
            'Records': len(animals_df),
            'Columns': len(animals_df.columns),
            'Primary Focus': 'Taxonomic Classification',
            'Key Strength': 'Complete taxonomic hierarchy'
        })
    
    if not plants_df.empty:
        comparison_metrics.append({
            'Dataset': 'Plants',
            'Records': len(plants_df),
            'Columns': len(plants_df.columns),
            'Primary Focus': 'Ecological Characteristics',
            'Key Strength': 'Rich ecological metadata'
        })
    
    if not biodiversity_df.empty:
        comparison_metrics.append({
            'Dataset': 'Biodiversity',
            'Records': len(biodiversity_df),
            'Columns': len(biodiversity_df.columns),
            'Primary Focus': 'Community Metrics',
            'Key Strength': 'Quantitative measures'
        })
    
    if comparison_metrics:
        comparison_df = pd.DataFrame(comparison_metrics)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization of dataset sizes
        fig = px.bar(comparison_df, x='Dataset', y='Records',
                    title="Dataset Size Comparison",
                    color='Dataset',
                    color_discrete_sequence=[theme_colors["animal"], theme_colors["plant"], theme_colors["biodiversity"]])
        st.plotly_chart(fig, use_container_width=True)


def main():
    build_enhanced_biology_dashboard()

if __name__ == "__main__":
    main()