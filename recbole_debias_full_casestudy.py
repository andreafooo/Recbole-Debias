from recbole_debias.quick_start import run_recbole_debias
import json
from recbole.utils.case_study import full_sort_topk

# from recbole.quick_start import load_data_and_model
import pandas as pd
import os
import yaml
import glob

from inherited.own_recbole_quickstart import load_data_and_model


# See evaluation metrics: https://recbole.io/docs/recbole/recbole.evaluator.metrics.html


config = "config_test.yaml"

with open(config, "r") as file:
    config_dict = yaml.safe_load(file)

OUTPUT_DIR = f"/Volumes/Forster Neu/Masterarbeit Data/{config_dict['dataset'].split('_')[0]}_dataset/recommendations/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def find_newest_model(directory):
    # Get list of all files in the directory
    files = glob.glob(os.path.join(directory, "*"))

    # Check if the directory is empty
    if not files:
        return None

    # Find the newest file based on creation time
    newest_file = max(files, key=os.path.getctime)
    return newest_file


def run_configurations(model, config):
    ### Run the RecBole model with specified configurations
    output_dict = run_recbole_debias(model=model, config_file_list=[config])

    ### Generating the recommendation list + other metadata
    # Directory to scan
    directory = "saved/"
    # Find the newest model file
    newest_model_file = find_newest_model(directory)

    if newest_model_file:
        model_file = newest_model_file.split("/")[-1]
    else:
        print("No model files found in the directory.")

    # Read user IDs from CSV
    df = pd.read_csv(
        f"dataset/{config_dict['dataset']}/{config_dict['dataset']}.train.inter",
        sep="\t",
    )
    user_ids = list(set(df["user_id:token"].values.tolist()))

    # Load model and data
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=f"saved/{model_file}",
    )

    print(f"model: {model}, dataset: {dataset}")

    error_count = 0
    recommendations = {}

    for uid in user_ids:
        try:
            # Convert external user ID to internal ID
            uid_series = dataset.token2id(dataset.uid_field, [uid])

            # Compute top-k recommendations
            topk_score, topk_iid_list = full_sort_topk(
                uid_series, model, test_data, k=15, device=config["device"]
            )

            if topk_score is None or topk_iid_list is None:
                print(f"Model did not return valid predictions for user {uid}")
                continue

            # print(f"TOP K SCORES for {uid} :")
            # print(topk_score)  # scores of top 10 items
            # print(topk_iid_list)  # internal id of top 10 items
            # Convert internal item IDs to external tokens
            external_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.cpu()
            )

            # Print or store recommendations
            # print(f"EXTERNAL TOKENS OF TOP k ITEMS FOR USER {uid} :")
            # print(external_item_list)

            # Computing the full scores
            # score = full_sort_scores(uid_series, model, test_data, device=config["device"])
            # print("Score of all items")
            # print(score)

            # print(
            #     score[0, dataset.token2id(dataset.iid_field, ["242", "302"])]
            # )  # score of item ['242', '302'] for user '196'.

            # Store recommendations in the desired format
            recommendations[uid] = [
                {"item_id": item_id, "score": score}
                for item_id, score in zip(
                    external_item_list.tolist(), topk_score.cpu().tolist()
                )
            ]

        except Exception as e:
            print(e)
            print(f"Error for user {uid}: {e}")
            error_count += 1
            continue

    # Print error rate
    print(
        f"The error count was {error_count}, which is {error_count/len(user_ids)*100:.2f}% of the total users."
    )

    model_file_cleaned = model_file.split(".")[0]

    FINAL_OUTPUT_DIR = (
        f"{OUTPUT_DIR}{config_dict['dataset']}-debias-{model_file_cleaned}/"
    )

    if not os.path.exists(FINAL_OUTPUT_DIR):
        os.makedirs(FINAL_OUTPUT_DIR)

    # Save the recommendations to a JSON file
    with open(f"{FINAL_OUTPUT_DIR}top_k_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=4)

    with open(f"{FINAL_OUTPUT_DIR}config.yaml", "w") as f:
        yaml.dump(config_dict, f, indent=4)

    with open(f"{FINAL_OUTPUT_DIR}general_evaluation.json", "w") as f:
        json.dump(output_dict, f, indent=4)

    # with open(f"{FINAL_OUTPUT_DIR}train_data.json", "w") as f:
    #     json.dump(train_data, f, indent=4)

    # with open(f"{FINAL_OUTPUT_DIR}valid_data.json", "w") as f:
    #     json.dump(valid_data, f, indent=4)

    # with open(f"{FINAL_OUTPUT_DIR}test_data.json", "w") as f:
    #     json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    run_configurations("PDA", config)
    run_configurations("MF", config)
    run_configurations("MACR", config)
