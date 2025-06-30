import json
import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common.utils import compute_style_control, compute_bootstrap_style_control, compute_bt, compute_bootstrap_bt

def get_bt_ratings(battle_data, anchor_model, anchor_rating=1000, num_bootstrap_samples=100, style_elements=None):
    """Compute Bradley-Terry ratings with optional style control elements"""
    if style_elements is None:
        bt_ratings = compute_bt(battle_data)
        offset_score = (anchor_rating - bt_ratings[anchor_model])
        bt_ratings += offset_score
        bt_ratings_bootstrap = compute_bootstrap_bt(battle_data, num_round=num_bootstrap_samples, offset=offset_score)
        style_coef_bootstrap = None
    else:
        bt_ratings, _ = compute_style_control(battle_data, style_elements=style_elements)
        offset_score = (anchor_rating - bt_ratings[anchor_model])
        bt_ratings += offset_score
        bt_ratings_bootstrap, style_coef_bootstrap = compute_bootstrap_style_control(
            battle_data, 
            style_elements=style_elements, 
            num_round=num_bootstrap_samples, 
            offset=offset_score
        )
    return bt_ratings_bootstrap, style_coef_bootstrap

def load_citation_data(file_path):
    """Load citation data and construct required DataFrame format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    return df

def prepare_battle_data(df):
    """Prepare battle data with conv_metadata and other required fields"""
    df['conv_metadata'] = pd.Series([{} for _ in range(len(df))])
    df['turn'] = 1
    
    for index in df.index:
        conv_metadata = df.loc[index, 'conv_metadata']
        
        # Add support, contradict, irrelevant counts
        conv_metadata['support_count_a'] = int(df.loc[index, 'support_count_a'])
        conv_metadata['support_count_b'] = int(df.loc[index, 'support_count_b'])
        conv_metadata['irrelevant_count_a'] = int(df.loc[index, 'irrelevant_count_a'])
        conv_metadata['irrelevant_count_b'] = int(df.loc[index, 'irrelevant_count_b'])
        
        # Add citation_count for filtering and analysis
        conv_metadata['citation_count_a'] = int(df.loc[index, 'total_count_a'])
        conv_metadata['citation_count_b'] = int(df.loc[index, 'total_count_b'])
        
        # Add response_length for length analysis
        conv_metadata['response_length_a'] = int(df.loc[index, 'response_length_a'])
        conv_metadata['response_length_b'] = int(df.loc[index, 'response_length_b'])
        
        df.at[index, 'conv_metadata'] = conv_metadata
    
    return df

def filter_battle_data(df):
    """Filter data following preference_analysis.ipynb logic"""
    # Filter battles with citations
    df = df[df['conv_metadata'].apply(lambda x: x['citation_count_a'] > 0 and x['citation_count_b'] > 0)]
    
    # Only keep battles where model_a or model_b wins
    df = df[df['winner'].isin(['model_a', 'model_b'])]
    
    return df.reset_index(drop=True)

def compute_citation_attribution_analysis(df):
    """Compute citation attribution analysis coefficient estimates"""
    
    # Define control elements
    CONTROL_ELEMENTS = ["support_count_a", "irrelevant_count_a"]
    CONTROL_ELEMENTS += ["support_count_b", "irrelevant_count_b"]
    
    bt_ratings_bootstrap, coef_bootstrap = get_bt_ratings(
        df, 
        anchor_model='GPT-4.1',
        anchor_rating=1000,
        num_bootstrap_samples=100,
        style_elements=CONTROL_ELEMENTS
    )
    
    # Coefficient names (remove _a and _b suffixes)
    coef_names = [s[:-2] for s in CONTROL_ELEMENTS[:(len(CONTROL_ELEMENTS)//2)]]
    
    # Compute confidence intervals and means
    lower = np.percentile(coef_bootstrap, 2.5, axis=0)
    upper = np.percentile(coef_bootstrap, 97.5, axis=0)
    mean = np.mean(coef_bootstrap, axis=0)
    
    coef_df = pd.DataFrame({
        'coef_name': coef_names,
        'mean': mean,
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_width': (upper - lower) / 2
    })
    
    coef_df = coef_df.sort_values(by='mean', ascending=True)
    return coef_df

def compute_citation_count_analysis(df):
    """Compute citation count analysis coefficient estimates"""
    
    CONTROL_ELEMENTS = ["citation_count_a", "citation_count_b"]
    
    bt_ratings_bootstrap, coef_bootstrap = get_bt_ratings(
        df, 
        anchor_model='GPT-4.1',
        anchor_rating=1000,
        num_bootstrap_samples=100,
        style_elements=CONTROL_ELEMENTS
    )
    
    coef_names = [s[:-2] for s in CONTROL_ELEMENTS[:(len(CONTROL_ELEMENTS)//2)]]
    
    lower = np.percentile(coef_bootstrap, 2.5, axis=0)
    upper = np.percentile(coef_bootstrap, 97.5, axis=0)
    mean = np.mean(coef_bootstrap, axis=0)
    
    coef_df = pd.DataFrame({
        'coef_name': coef_names,
        'mean': mean,
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_width': (upper - lower) / 2
    })
    
    coef_df = coef_df.sort_values(by='mean', ascending=True)
    return coef_df

def compute_response_length_analysis(df):
    """Compute response length analysis coefficient estimates"""
    
    CONTROL_ELEMENTS = ["response_length_a", "response_length_b"]
    
    bt_ratings_bootstrap, coef_bootstrap = get_bt_ratings(
        df, 
        anchor_model='GPT-4.1',
        anchor_rating=1000,
        num_bootstrap_samples=100,
        style_elements=CONTROL_ELEMENTS
    )
    
    coef_names = [s[:-2] for s in CONTROL_ELEMENTS[:(len(CONTROL_ELEMENTS)//2)]]
    
    lower = np.percentile(coef_bootstrap, 2.5, axis=0)
    upper = np.percentile(coef_bootstrap, 97.5, axis=0)
    mean = np.mean(coef_bootstrap, axis=0)
    
    coef_df = pd.DataFrame({
        'coef_name': coef_names,
        'mean': mean,
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_width': (upper - lower) / 2
    })
    
    coef_df = coef_df.sort_values(by='mean', ascending=True)
    return coef_df

def generate_length_control_leaderboard(df):
    """Generate elo leaderboard with length control"""
    
    style_elements = ['response_length_a', 'response_length_b']
    
    bt_ratings_bootstrap, _ = get_bt_ratings(
        df, 
        anchor_model='GPT-4.1',
        anchor_rating=1000,
        num_bootstrap_samples=100,
        style_elements=style_elements
    )
    
    model_order = list(bt_ratings_bootstrap.columns)
    
    model_rating_q025 = bt_ratings_bootstrap.quantile(0.025).round(2)
    model_rating_q975 = bt_ratings_bootstrap.quantile(0.975).round(2)
    bt_ratings = bt_ratings_bootstrap.mean().round(2)
    bt_var = bt_ratings_bootstrap.var().round(2)
    
    # Compute rankings
    ranking = {}
    for i, model_a in enumerate(model_order):
        ranking[model_a] = 1
        for j, model_b in enumerate(model_order):
            if i == j:
                continue
            if model_rating_q025[model_b] > model_rating_q975[model_a]:
                ranking[model_a] += 1
    
    num_battles = df["model_a"].value_counts().add(df["model_b"].value_counts(), fill_value=0)
    
    leaderboard_table = pd.DataFrame({
        "model": model_order,
        "rating": bt_ratings,
        "variance": bt_var,
        "rating_q975": model_rating_q975,
        "rating_q025": model_rating_q025,
        "num_battles": [num_battles.get(model, 0) for model in model_order],
        "final_ranking": [ranking[model] for model in model_order],
    })
    
    leaderboard_table = leaderboard_table.sort_values(by='rating', ascending=False)
    leaderboard_table = leaderboard_table.reset_index(drop=True)
    
    return leaderboard_table

def main():
    """Main function"""
    print("Starting Citation Attribution Analysis...")
    
    print("Loading citation_attribution.json data...")
    df = load_citation_data('data/citation_attribution.json')
    print(f"Loaded {len(df)} records")
    
    print("Preparing data...")
    df = prepare_battle_data(df)
    
    print("Filtering data...")
    df = filter_battle_data(df)
    print(f"Remaining {len(df)} records after filtering")
    
    if len(df) == 0:
        print("Warning: No data remaining after filtering, please check data format")
        return
    
    print("Computing Citation Attribution bootstrapped coefficient estimates...")
    attribution_coef_df = compute_citation_attribution_analysis(df)
    
    print("Computing Citation Count bootstrapped coefficient estimates...")
    count_coef_df = compute_citation_count_analysis(df)
    
    print("Computing Response Length bootstrapped coefficient estimates...")
    length_coef_df = compute_response_length_analysis(df)
    
    print("Generating Length Control Leaderboard...")
    leaderboard_df = generate_length_control_leaderboard(df)
    
    result_dict = {
        "citation_attribution": {},
        "citation_count": {},
        "response_length": {}
    }
    
    # Convert Citation Attribution results to dict format
    for _, row in attribution_coef_df.iterrows():
        result_dict["citation_attribution"][row['coef_name']] = {
            'mean': float(row['mean']),
            'ci_lower': float(row['ci_lower']),
            'ci_upper': float(row['ci_upper']),
            'ci_width': float(row['ci_width'])
        }
    
    # Convert Citation Count results to dict format
    for _, row in count_coef_df.iterrows():
        result_dict["citation_count"][row['coef_name']] = {
            'mean': float(row['mean']),
            'ci_lower': float(row['ci_lower']),
            'ci_upper': float(row['ci_upper']),
            'ci_width': float(row['ci_width'])
        }
    
    # Convert Response Length results to dict format
    for _, row in length_coef_df.iterrows():
        result_dict["response_length"][row['coef_name']] = {
            'mean': float(row['mean']),
            'ci_lower': float(row['ci_lower']),
            'ci_upper': float(row['ci_upper']),
            'ci_width': float(row['ci_width'])
        }
    
    # Save results to JSON file
    output_file = 'results/preference_analysis_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print("\nCitation Attribution Results Summary:")
    print(attribution_coef_df.to_string(index=False))
    print("\nCitation Count Results Summary:")
    print(count_coef_df.to_string(index=False))
    print("\nResponse Length Results Summary:")
    print(length_coef_df.to_string(index=False))
    
    # Save Length Control Leaderboard to CSV file
    leaderboard_output_file = 'results/length_control_leaderboard.csv'
    leaderboard_df.to_csv(leaderboard_output_file, index=False, encoding='utf-8')
    print(f"\nLength Control Leaderboard saved to {leaderboard_output_file}")
    print("Length Control Leaderboard:")
    print(leaderboard_df.to_string(index=False))
    
    # Save detailed results in DataFrame format
    detailed_results = {
        'citation_attribution': attribution_coef_df.to_dict('records'),
        'citation_count': count_coef_df.to_dict('records'),
        'response_length': length_coef_df.to_dict('records')
    }
    
    with open('results/preference_analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print("Detailed results saved to preference_analysis_result.json")

if __name__ == "__main__":
    main() 