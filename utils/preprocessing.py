"""Module containg all necesary steps for the preprocessing of the data
"""

import os

from tqdm import tqdm

import pandas as pd


def create_annotations_file_from_raw_labels(path_to_raw_labels: str, output_path: str) -> None:
    list_of_labels = list()
    for label_file in tqdm(os.listdir(path_to_raw_labels)):
        label = label_file_to_label(os.path.join(path_to_raw_labels, label_file))
        list_of_labels.append(label)

    label_df = pd.DataFrame(list_of_labels)

    label_df.to_csv(
        output_path,
        index=None
    )
    return

def label_file_to_label(path: str) -> pd.Series:
    img_base_name = os.path.basename(path).replace(".txt", "")
    try:
        label_df = pd.read_csv(
            path,
            delimiter=" ",
            header=None,
        )
    except pd.errors.EmptyDataError:
        label_df = pd.DataFrame(data={0:[]})

    value_list = list()
    for c in range(3):
        label_row = label_df[label_df[0]==c]
        if len(label_row) != 0:
            value_list += label_row.values.flatten().tolist()[1:]
        else:
            value_list += [-1] * 4

    label_list = [img_base_name] + value_list

    label = pd.Series(
        data=label_list,
        index=["img_id", "x_0", "y_0", "w_0", "h_0", "x_1", "y_1", "w_1", "h_1", "x_2", "y_2", "w_2", "h_2"]
    )

    return label


if __name__ == "__main__":
    pass
    # label_folder = "/home/thomas/projects/cv_project_template/data/brain_tumor_dataset/raw/train/labels/"
    # label_output_path = "/home/thomas/projects/cv_project_template/data/brain_tumor_dataset/preprocessed/labels/labels.csv"
    # create_annotations_file_from_raw_labels(
    #     label_folder,
    #     label_output_path
    # )
