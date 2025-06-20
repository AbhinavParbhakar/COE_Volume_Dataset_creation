import pandas as pd
from tqdm import tqdm

class Neighbors:
    def __init__(self,threshold,dataset_file):
        self.road_df = pd.read_csv('./data/excel_files/Road connection.csv')
        self.point_df = pd.read_csv(dataset_file)
        self.point_df['Point ID'] = pd.Series([i+1 for i in range(self.point_df.shape[0])])
        self.threshold = threshold
        self.graph = {}
        
        for _, row in self.road_df.iterrows():
            r1 = row['Road ID']
            r2 = row['Link ID']
            self.graph.setdefault(r1, set()).add(r2)
            self.graph.setdefault(r2, set()).add(r1)

        self.road_lengths = {}
        for _, row in self.road_df.iterrows():
            r1 = row['Road ID']
            r2 = row['Link ID']
            len_r1 = row['Road Length']
            len_r2 = row['Link Length']
            if r1 not in self.road_lengths:
                self.road_lengths[r1] = len_r1
            if r2 not in self.road_lengths:
                self.road_lengths[r2] = len_r2

        self.road_to_point = {}
        for _, row in self.point_df.iterrows():
            road = row['Road Class FID']
            point = row['Point ID']
            self.road_to_point[road] = point

    def find_data_point(self,current_road, original_road, visited, current_distance):
        if current_road in self.road_to_point and current_road != original_road:
            candidate_distance = current_distance + 0.5 * self.road_to_point.get(current_road, 0)
            if candidate_distance <= self.threshold:
                return {(self.road_to_point[current_road], candidate_distance)}
            else:
                return set()

        additional_length = self.road_lengths.get(current_road, 0)
        new_distance = current_distance + additional_length
        if new_distance > self.threshold:
            return set()

        results = set()
        visited.add(current_road)

        # 遍历当前路段的所有邻接路段
        if current_road in self.graph:
            for neighbor in self.graph[current_road]:
                if neighbor not in visited:
                    results = results.union(
                        self.find_data_point(neighbor, original_road, visited, new_distance)
                    )
        return results

    def generate_neighbors_csv(self,save_name):
        adjacency = {}
        for _, row in tqdm(self.point_df.iterrows(), total=len(self.point_df)):
            point_id = row['Point ID']
            road = row['Road Class FID']
            neighbors = set()
            initial_distance = 0.5 * self.road_lengths.get(road, 0)
            if road in self.graph:
                for neighbor_road in self.graph[road]:
                    results = self.find_data_point(neighbor_road, road, set(), initial_distance)
                    neighbors = neighbors.union(results)
            adjacency[point_id] = neighbors

        adj_pairs = {}
        for point, neighbor_set in adjacency.items():
            for neighbor, distance in neighbor_set:
                pair = tuple(sorted([point, neighbor]))
                if pair in adj_pairs:
                    adj_pairs[pair] = min(adj_pairs[pair], distance)
                else:
                    adj_pairs[pair] = distance

        adj_list = [{'Point1': p1, 'Point2': p2, 'Distance': d} for (p1, p2), d in adj_pairs.items()]
        adj_df = pd.DataFrame(adj_list)
        adj_df.to_csv(f'./data/excel_files/{save_name}.csv', index=False)

if __name__ == "__main__":
    print("Hello")