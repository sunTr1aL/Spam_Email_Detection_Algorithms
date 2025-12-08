import os
import glob
import pandas as pd

from config import OUTPUT_DIR


def calculate_weighted_average(group_df, group_name):
    if group_df["size"].sum() == 0:
        return None

    total_size = group_df["size"].sum()
    group_df["Weight"] = group_df["size"] / total_size

    weighted_row = {
        "Dataset": "combined",
        "Split": f"weighted_{group_name}",
        "Model": "RoBERTa",
        "Acc": (group_df["Acc"] * group_df["Weight"]).sum(),
        "Prec": (group_df["Prec"] * group_df["Weight"]).sum(),
        "Recall": (group_df["Recall"] * group_df["Weight"]).sum(),
        "F1": (group_df["F1"] * group_df["Weight"]).sum(),
    }
    return weighted_row


def generate_summary():
    print(f"Summary Generation: Searching in {OUTPUT_DIR}...")

    search_path = os.path.join(OUTPUT_DIR, "**", "metrics_*.csv")
    file_paths = glob.glob(search_path, recursive=True)

    if not file_paths:
        print("No metrics files found. Please run run.py first.")
        return

    all_results = []

    for path in file_paths:
        try:
            df = pd.read_csv(path)
            if df.empty or "size" not in df.columns:
                print(f"Skipping {path}: Missing required 'size' column.")
                continue

            tag = os.path.basename(path).replace("metrics_", "").replace(".csv", "")
            if "train_" in tag and "test_" in tag:
                dataset = tag.replace("train_", "").replace("test_", "")
                split_type = "cross_dataset"
            elif "ratio_" in tag:
                split_strs = tag.split("_")
                dataset = split_strs[0]
                if split_strs[1] == "assassin":
                    split_type = tag.split("_", 2)[2]  # e.g., ratio_0.1
                else:
                    split_type = tag.split("_", 1)[1]
            else:
                parts = tag.split("_")
                dataset = parts[0]
                split_type = parts[1]  # e.g., random, time

            all_results.append(
                {
                    "Dataset": dataset,
                    "Split": split_type,
                    "Model": "RoBERTa",
                    "Acc": df["accuracy"].iloc[0],
                    "Prec": df["precision"].iloc[0],
                    "Recall": df["recall"].iloc[0],
                    "F1": df["f1"].iloc[0],
                    "size": df["size"].iloc[0],
                    "Exp_Group": split_type,
                }
            )

        except Exception as e:
            print(f"Error processing file {path}: {e}")
            continue

    if not all_results:
        print("No valid results were successfully processed.")
        return

    results_df = pd.DataFrame(all_results)
    combined_results = []

    for group_name, group_df in results_df.groupby("Exp_Group"):
        combined_row = calculate_weighted_average(group_df, group_name)
        if combined_row:
            combined_results.append(combined_row)

    combined_df = pd.DataFrame(combined_results)

    final_cols = ["Dataset", "Split", "Model", "Acc", "Prec", "Recall", "F1"]

    final_df = pd.concat(
        [results_df[final_cols], combined_df[final_cols]], ignore_index=True
    )

    final_df = final_df.sort_values(by=["Dataset", "Split"]).reset_index(drop=True)

    output_file = os.path.join(OUTPUT_DIR, "results.csv")
    final_df.to_csv(output_file, index=False, float_format="%.4f")

    print(f"\nSummary complete.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    generate_summary()
