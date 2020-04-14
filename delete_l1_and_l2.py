import pandas as pd

def del_1_and_2s(path):
    csv = pd.read_csv(path)
    csv = csv[(csv["l_norm"] == "i")]
    csv.to_csv(path, index=False)

if __name__ == "__main__":
    datasets=["mnist", "cifar", "caltechSilhouettes", "GTSRB", "sign-language"]
    for i in datasets:
        del_1_and_2s(f"v10/{i}]/upper_bound.csv")