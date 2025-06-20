import pandas as pd
import numpy as np


class AdjacencyMatrix:
    def __init__(self,file_path):
        self.df = pd.read_csv(file_path)

    def generate_adj_matrix(self,num_points,save_name):
        points = list([i + 1 for i in range(num_points)])

        adj_matrix = np.full((num_points, num_points), np.inf)
        np.fill_diagonal(adj_matrix, 0)

        for _, row in self.df.iterrows():
            p1 = int(row['Point1'])
            p2 = int(row['Point2'])
            distance = row['Distance']
            i, j = p1 - 1, p2 - 1
            adj_matrix[i, j] = distance
            adj_matrix[j, i] = distance

        print(adj_matrix.shape)
        adj_df = pd.DataFrame(adj_matrix, index=points, columns=points)
        adj_df.to_csv(f'./data/excel_files/{save_name}.csv', index=False)
