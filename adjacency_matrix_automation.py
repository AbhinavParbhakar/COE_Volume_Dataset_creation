from Find_neighbors import Neighbors
from Adjacency_matrix import AdjacencyMatrix
import pandas as pd

if __name__ == "__main__":
    max_range = 5000 # meters
    dataset_file = './data/excel_files/Final_Dataset.csv'
    df = pd.read_csv(dataset_file)
    num_points = df.shape[0]
    threshold_skips = 500 # meters
    start_distance = 1000 # meters
    thresholds = [i for i in range(start_distance,max_range + 1,threshold_skips)]
    for threshold in thresholds:
        neighbor_creation = Neighbors(threshold=threshold,dataset_file=dataset_file)
        road_links_name = f'Road_links_{threshold}_meters'
        adj_matrix_links_name = f'Adjacency Matrix {threshold}_meters'
        neighbor_creation.generate_neighbors_csv(save_name=road_links_name)
        # adj_matrix = AdjacencyMatrix(file_path=f'./data/excel_files/{road_links_name}.csv')
        # adj_matrix.generate_adj_matrix(num_points=num_points,save_name=adj_matrix_links_name)