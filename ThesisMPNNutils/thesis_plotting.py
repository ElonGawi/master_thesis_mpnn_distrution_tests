import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from labellines import labelLines
from matplotlib.patches import Patch
from netcal.metrics import ECE
from sklearn.calibration import calibration_curve


# # e.g.
# # dataset = "train"
# # experiment_num = "1"
# # folder_path =  "/data/leuven/368/vsc36835/thesis/MPNN/ProteinMPNN/thesis/feather_files"
# # class_name = "-"
def save_per_class_probs_and_ground_truth_to_file(
    experiment_num,
    dataset,
    class_name,
    y_prob_class,
    y_true_class,
    folder_path="/data/leuven/368/vsc36835/thesis/MPNN/ProteinMPNN/thesis/feather_files",
):

    feather_file_name = f"y_prob_y_ground_truth_experiment_{experiment_num}_dataset_type_{dataset}_class_{class_name}.parquet"
    feather_file_path = os.path.join(folder_path, feather_file_name)

    if os.path.exists(feather_file_path):
        raise Exception("File exists!")

    df_class = pd.DataFrame({"y_prob": y_prob_class, "y_true": y_true_class})

    # df_class.to_feather(feather_file_path)
    df_class.to_parquet(feather_file_path, engine='pyarrow')




def plot_reliablity(
    all_prob_matrix,
    all_seq,
    alphabet="ACDEFGHIKLMNPQRSTVWY-",
    n_bins=10,
    ece_threshold=0.05,
    output_folder=None,
    create_feather_files=True,
    file_name_dict={"expriment_num": "Unknown", "dataset": "Unknown"},
):
    amino_acids_num = len(alphabet)
    seq_length = len(all_seq)

    outliers = []
    backgrounds = []
    bg_y_true_pooled = []
    bg_y_prob_pooled = []

    # Initialize the official NetCal ECE evaluator
    ece_metric = ECE(bins=n_bins)

    # --- 1. Process Data & Calculate Metrics ---
    for i in range(amino_acids_num):
        amino_acid_letter = alphabet[i]

        # Generate True/False binary array for the current amino acid
        y_true_class = np.zeros(seq_length).astype(int)
        idx_in_original_seq = find_occurences_in_seq(
            seq=all_seq, amino_acid=amino_acid_letter
        )
        for idx in idx_in_original_seq:
            y_true_class[idx] = 1

        y_prob_class = all_prob_matrix[:, i]

        #### save files
        if output_folder:
            save_per_class_probs_and_ground_truth_to_file(
                experiment_num=file_name_dict["expriment_num"],
                dataset=file_name_dict["dataset"],
                class_name=str(amino_acid_letter),
                y_prob_class=y_prob_class,
                y_true_class=y_true_class,
                folder_path=output_folder,
            )
        ####

        # Calculate ECE using NetCal
        ece_score = ece_metric.measure(y_prob_class, y_true_class)

        # Calculate curve coordinates
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_class, y_prob_class, n_bins=n_bins, strategy="uniform"
        )

        data_dict = {
            "letter": amino_acid_letter,
            "ece": ece_score,
            "x": mean_predicted_value,
            "y": fraction_of_positives,
        }

        # Sort into outliers vs well-calibrated backgrounds
        if ece_score > ece_threshold:
            outliers.append(data_dict)
        else:
            backgrounds.append(data_dict)
            bg_y_true_pooled.append(y_true_class)
            bg_y_prob_pooled.append(y_prob_class)

    # --- 2. Setup Plot ---
    fig, ax = plt.subplots(figsize=(4, 4))

    # Perfectly calibrated diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.5, zorder=1)

    # --- 3. Plot the Faint Background Bundle ---
    for item in backgrounds:
        if len(item["x"]) > 0:
            ax.plot(
                item["x"],
                item["y"],
                marker="None",
                linestyle="-",
                color="lightgray",
                alpha=0.5,
                linewidth=1.0,
                zorder=2,
            )

    # --- 4. Calculate and Plot the AVERAGE Baseline Curve ---
    if bg_y_true_pooled:
        pooled_true = np.concatenate(bg_y_true_pooled)
        pooled_prob = np.concatenate(bg_y_prob_pooled)
        avg_y, avg_x = calibration_curve(
            pooled_true, pooled_prob, n_bins=n_bins, strategy="uniform"
        )

        #### save files
        if output_folder:
            save_per_class_probs_and_ground_truth_to_file(
                experiment_num=file_name_dict["expriment_num"],
                dataset=file_name_dict["dataset"],
                class_name="ALL_residues",
                y_prob_class=pooled_prob,
                y_true_class=pooled_true,
                folder_path=output_folder,
            )
        ####

        # Plotted without a label so it doesn't get inline text
        ax.plot(
            avg_x,
            avg_y,
            marker="s",
            markersize=5,
            linestyle="-",
            color="#404040",
            alpha=0.9,
            linewidth=2.5,
            zorder=3,
        )

    # --- 5. Plot the Highlighted Outliers ---
    colors = [
        "#e6194B",
        "#4363d8",
        "#f58231",
        "#3cb44b",
        "#911eb4",
        "#f032e6",
        "#bfef45",
        "#42d4f4",
    ]
    for idx, item in enumerate(outliers):
        if len(item["x"]) > 0:
            c = colors[idx % len(colors)]
            ax.plot(
                item["x"],
                item["y"],
                marker="o",
                markersize=5,
                linestyle="-",
                color=c,
                alpha=0.9,
                linewidth=2.0,
                zorder=4,
                label=f" {item['letter']} ",
            )

    # --- 6. Generate the Custom Legend (MUST be done before inline labels) ---
    # --- 6. Generate the Custom Legend (Patch method to bypass Python 3.14 bug) ---
    legend_elements = []

    if bg_y_true_pooled:
        if len(outliers) == 0:
            avg_legend_text = "All residues"
        else:
            avg_legend_text = f" All residues with (ECE ≤ {ece_threshold})"

        legend_elements.append(
            Patch(facecolor="#404040", edgecolor="#404040", label=avg_legend_text)
        )

    if len(outliers) > 0:
        legend_elements.append(
            Patch(
                facecolor="gray",
                edgecolor="black",
                label=f"Outlier Classes (ECE > {ece_threshold})",
            )
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgray",
            fontsize=11,
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgray",
            fontsize=11,
        )

    # --- 7. Apply Inline Labels to Outliers ---
    lines_to_label = [
        line
        for line in ax.get_lines()
        if line.get_label() and not line.get_label().startswith("_")
    ]
    labelLines(
        lines_to_label,
        align=False,
        color="black",
        fontsize=12,
        fontweight="bold",
        zorder=5,
    )

    # --- 8. Professional Formatting ---
    ax.set_title("Multiclass Reliability Diagram (ProteinMPNN)", fontsize=14, pad=15)
    ax.set_xlabel("ProtienMPNN Confidence", fontsize=12)
    ax.set_ylabel("Accuracy)", fontsize=12)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="-", alpha=0.3)

    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"class_realibility_diag_10_bins_expriment_{file_name_dict["expriment_num"]}_dataset_{file_name_dict["dataset"]}_{timestamp}.svg"

    # Save the file
    plt.savefig(os.path.join(output_folder, filename), format="svg", bbox_inches="tight")
    print(f"Plot saved to: {filename}")
    plt.show()




def find_occurences_in_seq(seq, amino_acid):
    return [index for index, character in enumerate(seq) if character == amino_acid]

def get_metrics_amino_acid(
    all_prob_matrix, all_seq, alphabet="ACDEFGHIKLMNPQRSTVWY-", n_bins=10
):

    amino_acids_num = len(alphabet)
    seq_length = len(all_seq)

    # Initialize the official NetCal ECE evaluator
    ece_metric = ECE(bins=n_bins)

    # ACE evaluator
    # ace_metric = ACE(bins=n_bins)

    return_dict = {}

    for i in range(amino_acids_num):
        amino_acid_metrics = {}
        amino_acid_letter = alphabet[i]

        # Generate True/False binary array for the current amino acid
        y_true_class = np.zeros(seq_length).astype(int)
        idx_in_original_seq = find_occurences_in_seq(
            seq=all_seq, amino_acid=amino_acid_letter
        )
        for idx in idx_in_original_seq:
            y_true_class[idx] = 1

        y_prob_class = all_prob_matrix[:, i]

        # uniform bin count
        uniform_bin_counts, uniform_bin_edges = np.histogram(
            y_prob_class, bins=n_bins, range=(0, 1)
        )

        # Calculate ECE using NetCal
        ece_score = ece_metric.measure(y_prob_class, y_true_class)

        # Calculate ACE using NetCal
        # ace_score = ace_metric.measure(y_prob_class, y_true_class)

        amino_acid_metrics["ece_score"] = ece_score
        # amino_acid_metrics["ace_score"] = ace_score

        amino_acid_metrics["uniform_bin_counts"] = uniform_bin_counts
        amino_acid_metrics["uniform_bin_edges"] = uniform_bin_edges

        return_dict[amino_acid_letter] = amino_acid_metrics

    return return_dict




def save_dict_to_file(dict_to_save, output_folder, file_name_start, file_name_dict={"expriment_num": "Unknown", "dataset": "Unknown"}):
    filename = f"{file_name_start}_expriment_{file_name_dict["expriment_num"]}_dataset_{file_name_dict["dataset"]}.json"
    full_path = os.path.join(output_folder, filename)
    
    if os.path.exists(full_path):
        raise Exception("File exists!")

    with open(os.path.join(full_path), "w") as f:
        f.write(str(dict_to_save))



def get_metric_avg(metrics_amino_acids):
    all_ece = []    
    num_classes = len(metrics_amino_acids.keys())

    for _, v in metrics_amino_acids.items():
        all_ece.append(v["ece_score"])

    all_ece = np.array(all_ece)

    return {"mean_ece": all_ece.mean()}


def get_worst_ece(metrics_amino_acids):

    first_res = list(metrics_amino_acids.keys())[0]
    # initialize with the first one
    highest_ece = first_res, metrics_amino_acids[first_res]["ece_score"]

    for k, v in metrics_amino_acids.items():
        if metrics_amino_acids[k]["ece_score"] > highest_ece[1]:
            highest_ece = k, metrics_amino_acids[k]["ece_score"]

    return {"highest_ece": highest_ece}



import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_aa_confidence_histograms(
    metrics_amino_acids,
    output_folder=None,
    file_name_dict={"expriment_num": "Unknown", "dataset": "Unknown"},
):
    """
    baically this is trying to check if there are diffrent freqeuncies of certain confidence ranges for the diffrent amino acids
    """
    # assuming that all the reisdues have the same bin count
    n_bins = len(metrics_amino_acids["A"]["uniform_bin_counts"])

    # 1. Prepare the residues (alphabetical, but put '-' at the end)
    residues = sorted([k for k in metrics_amino_acids.keys() if k != "-"])
    if "-" in metrics_amino_acids:
        residues.append("-")

    # 2. Setup the grid (5 columns, rows as needed)
    num_res = len(residues)
    cols = 5
    rows = (num_res + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.5), sharex=True)
    axes = axes.flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.0 / n_bins

    for i, res in enumerate(residues):
        ax = axes[i]
        counts = metrics_amino_acids[res]["uniform_bin_counts"]
        ece = metrics_amino_acids[res]["ece_score"]

        # Create bars
        # Using log=True directly in bar() or ax.set_yscale('log')
        ax.bar(
            bin_centers,
            counts,
            width=width,
            color="royalblue",
            edgecolor="black",
            alpha=0.7,
            log=True,
        )

        # Residue specific formatting
        ax.set_title(
            f"Residue: {res}\n$ECE = {ece:.4f}$",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, which="both", ls="-", alpha=0.1)

        # Only show labels on the edges to keep it clean
        if i % cols == 0:
            ax.set_ylabel("Log Count")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Confidence")

    # 3. Clean up empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Confidence Distributions per Amino Acid (log scale)", fontsize=20, y=1.02
    )
    plt.tight_layout()

    if output_folder:
        # 4. Save with requested timestamp format
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"historgams_preds_expriment_{file_name_dict['expriment_num']}_dataset_{file_name_dict['dataset']}_{timestamp}.svg"
        full_path = os.path.join(output_folder, filename)
        if os.path.exists(full_path):
            raise Exception("File Exist")
        plt.savefig(full_path, format="svg", bbox_inches="tight")

        print(f"Comparison plot saved to: {filename}")
    plt.show()




from sklearn.metrics import log_loss
import numpy as np


def get_perplexity_sklearn(y_true_indices, y_prob_matrix):
    """
    y_true_indices: Array of shape (N,) containing the true AA index (0-19)
    y_prob_matrix: Array of shape (N, 20) containing the model probabilities
    """
    # 1. Calculate the Cross-Entropy (Log Loss)
    # labels parameter is important if some AAs are missing in your sample
    cross_entropy = log_loss(
        y_true_indices, y_prob_matrix, labels=np.arange(y_prob_matrix.shape[1])
    )

    # 2. Convert to Perplexity
    perplexity = np.exp(cross_entropy)

    return perplexity

def convert_ground_truth_to_indicies(all_seq):
    # convert the ground truth to the index
    alphabet = "ACDEFGHIKLMNPQRSTVWY-"
    return [alphabet.index(s) for s in all_seq]
    

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import top_k_accuracy_score
from datetime import datetime


def plot_simple_topk(
    y_true_indices,
    y_prob_matrix,
    output_folder=None,
    file_name_dict={"expriment_num": "Unknown", "dataset": "Unknown"},
):
    """
    Calculates and plots Top-K accuracy for the whole dataset.
    """
    # 1. Calculate accuracies for K = 1, 2, 3, 4, 5
    ks = [1, 2, 3, 4, 5]
    accuracies = [top_k_accuracy_score(y_true_indices, y_prob_matrix, k=k) for k in ks]

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(ks, accuracies, color="royalblue", edgecolor="black", alpha=1.0)
    ax.set_axisbelow(True)

    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.2%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Formatting
    ax.set_title("Top-K Accuracy", fontsize=14, pad=15)
    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.1)  # Leave space for labels
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if output_folder:
        # 3. Save with requested timestamp: year_month_day_hour_minues_seconds.svg
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"top_k_expriment_{file_name_dict['expriment_num']}_dataset_{file_name_dict['dataset']}_{timestamp}.svg"
        full_path = os.path.join(output_folder, filename)
        if os.path.exists(full_path):
            raise Exception("File Exist")
        plt.savefig(full_path, format="svg", bbox_inches="tight")

        print(f"Plot saved as: {filename}")
    plt.show()


def create_and_save_all_metrics_for_expriment(all_prob_matrix, all_seq, output_folder, file_name_dict):
    if os.path.exists(output_folder):
        raise Exception("outptu Folder already exists! ")

    os.makedirs(output_folder)
    # plot and 
    plot_reliablity(
        all_prob_matrix,
        all_seq,
        alphabet="ACDEFGHIKLMNPQRSTVWY-",
        n_bins=10,
        ece_threshold=0.05,
        file_name_dict=file_name_dict,
        output_folder=output_folder,
        create_feather_files=True
    )

    metrics_amino_acids = get_metrics_amino_acid(all_prob_matrix, all_seq)
    save_dict_to_file(
        metrics_amino_acids, output_folder=output_folder, file_name_start="class_metrics", file_name_dict=file_name_dict
    )

    extra_metrics_dict = {}
    extra_metrics_dict.update(get_metric_avg(metrics_amino_acids))
    extra_metrics_dict.update(get_worst_ece(metrics_amino_acids))
    extra_metrics_dict.update({"all_prob_matrix_shape": str(all_prob_matrix.shape)})


    plot_aa_confidence_histograms(
        metrics_amino_acids,
        output_folder=output_folder,
        file_name_dict=file_name_dict
    )


    y_true_indices = convert_ground_truth_to_indicies(all_seq)
    y_prob_matrix = all_prob_matrix
    extra_metrics_dict.update({"preplexity": get_perplexity_sklearn(y_true_indices, y_prob_matrix)})


    plot_simple_topk(y_true_indices, y_prob_matrix, file_name_dict=file_name_dict, output_folder=output_folder)

    save_dict_to_file(
        extra_metrics_dict, output_folder=output_folder, file_name_start="extra_metrics", file_name_dict=file_name_dict
    )


