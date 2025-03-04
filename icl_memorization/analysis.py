import pickle
import traceback
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
 
wandb_group_name = "memo_feb24_zipf"
# Filter runs by group name
# Initialize lists to store data for the heatmap
n_heads_list = []
n_layers_list = []
accuracy_list = []

item=0
runs = glob(f"./cache/{wandb_group_name}/{wandb_group_name}_*.pkl")

import json
from collections import defaultdict
runs=["./cache/memo_feb24_zipf/memo_feb24_zipf_transformer_K_1000_L_100_hidden_8_1738918219.624416.pkl"]
for run in runs :
    # Extract model configurations
    print(run)
    # try: 
    with open(run, "rb") as f:
        record = pickle.load(f)
    # record = json.loads(run)
    print(record.keys(), len(record["logs"]), record["logs"][0].keys())
    # except Exception as e: 
    #     print (traceback.format_exc())
    #     continue
    
    # Extract model configurations
    acc_vs_presentations = defaultdict(list)
    for i, l in enumerate(record["logs"]):
        test = l.get("test_metrics")
        if len(test) > 0:
            print(test.keys())
            test = pd.DataFrame(test)
            #display(test)
            pivot = test.pivot_table(index="length", columns="sequence_rank", values="logsoftmaxloss")
            #display(pivot)
            plt.figure(figsize=(12,8))
            sns.heatmap(pivot, annot=False, vmin=0, vmax=1)
            plt.xlabel("Sequence Rank")
            plt.ylabel("Position in Sequence")
            plt.title("Log Softmax Loss vs. Sequence Rank vs. Position in Sequence")
            plt.tight_layout()
            plt.savefig(f"./cache/zipf_figs/logsoftmaxloss_vs_sequence_rank_vs_position_in_sequence_epoch_{i}.png")
            plt.show()
            pivot = test.pivot_table(index="length", columns="sequence_rank", values="accuracy")
            #display(pivot)
            plt.figure(figsize=(12,8))
            sns.heatmap(pivot, annot=False)
            plt.xlabel("Sequence Rank")
            plt.ylabel("Position in Sequence")
            plt.title("Accuracy vs. Sequence Rank vs. Position in Sequence")
            plt.savefig(f"./cache/zipf_figs/accuracy_vs_sequence_rank_vs_position_in_sequence_epoch_{i}.png")
            plt.show()
            K = record["args"]["K"]
            p = np.array([1/(i+1) for i in range(K)])
            p /= np.sum(p)
            expected_number_of_presentations = l.get ("num_apppearances") 
            expected_number_of_presentations += (np.zeros_like(expected_number_of_presentations)+1e-10)
            # display (test)
            # average over sequence rank, and plot logsoftmaxloss vs. expected number of presentations 
            avg_test = test.groupby("sequence_rank").mean()
             
            # loss_at_20 = test[test["length"] == 20]["logsoftmaxloss"] 
            avg_loss = avg_test["logsoftmaxloss"]
            accuracy_vs_presentations = test.groupby("sequence_rank").mean()["accuracy"].to_list()
             
            # print(accuracy_at_20)
            # display (loss_at_20)
            loss_vs_expected_number_of_presentations = pd.DataFrame({
                "expected_number_of_presentations": expected_number_of_presentations,
                "logsoftmaxloss": avg_loss,
                "accuracy_vs_presentations": accuracy_vs_presentations, 
            })
            acc_vs_presentations["number_of_presentations"].extend(expected_number_of_presentations)
            acc_vs_presentations["logsoftmaxloss"].extend(avg_loss)
            acc_vs_presentations["accuracy_vs_presentations"].extend(accuracy_vs_presentations)
            acc_vs_presentations["rank"].extend(range(K))
            sns.scatterplot (x = "expected_number_of_presentations", y = "logsoftmaxloss", data = loss_vs_expected_number_of_presentations)
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("Expected Number of Presentations")
            plt.ylabel("Log Loss")
            plt.tight_layout()
            plt.savefig(f"./cache/zipf_figs/logsoftmaxloss_vs_expected_number_of_presentations_epoch_{i}.png")
            plt.show()
            sns.scatterplot (x = "expected_number_of_presentations", y = "accuracy_vs_presentations", data = loss_vs_expected_number_of_presentations)
            plt.xscale("log")
            plt.xlabel("Expected Number of Presentations")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f"./cache/zipf_figs/accuracy_vs_expected_number_of_presentations_epoch_{i}.png")
            plt.show()
        else:
            print ("epoch", i, "no test_metrics")

    plt.figure(figsize=(12,8))
    # plot logsoftmaxloss vs. number of presentations
    sns.lineplot(x = "number_of_presentations", y = "logsoftmaxloss", hue = "rank", data = acc_vs_presentations)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Expected Number of Presentations")
    plt.ylabel("Log Loss")
    plt.tight_layout()
    plt.savefig(f"./cache/zipf_figs/logsoftmaxloss_vs_number_of_presentations_per_rank.png")
    plt.show()
    plt.figure(figsize=(12,8))
    # plot accuracy vs. number of presentations
    sns.lineplot(x = "number_of_presentations", y = "accuracy_vs_presentations", hue = "rank", data = acc_vs_presentations)
    plt.xscale("log")
    plt.xlabel("Expected Number of Presentations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"./cache/zipf_figs/accuracy_vs_number_of_presentations_per_rank.png")
    plt.show()