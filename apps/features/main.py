from extract import get_fightsnapshots_df, unnest_raw_df
from build_features import FightFeatures

def main():
    # 1. Extract and unnest raw data into 3 dfs that can be grouped by root_fight_id
    raw_df = get_fightsnapshots_df()
    fights_df, prior_fights_df, prior_rounds_df = unnest_raw_df(raw_df)

    # 2. Generate features
    features = FightFeatures(fights_df, prior_fights_df, prior_rounds_df)
    features.root_fights_meta_df()
    print(features.final_df)

if __name__ == "__main__":
    main()
