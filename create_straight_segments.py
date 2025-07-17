import geopandas as gpd
from shapely import LineString, force_2d, line_interpolate_point, Point
import numpy as np
from shapely.wkt import loads
from shapely.ops import split, snap, transform, nearest_points
import pandas as pd


def generate_split(line:LineString,point:Point)->tuple[LineString,LineString]:
    """
    Generate two splits of the given line based on the point
    
    ### Parameters
        line: **shapely.LineString**
        point: **shapely.Point**
        
    ### Returns
        A **tuple[shapely.LineString,shapely.LineString]** object containing the left and right components of the original segment
        
    """
    line_coordinates = [Point(coordinate) for coordinate in list(line.coords)]
    assert len(line_coordinates) >= 2, "LineString must have at least two points before being split"
    i = 1
    insert_index = 0
    lowest_dis = 1000
    while i <= len(line_coordinates) - 1:
        connecting_line = LineString([line_coordinates[i-1],line_coordinates[i]])
        distance = connecting_line.distance(point)
        if distance < lowest_dis:
            lowest_dis = distance
            insert_index = i
        
        i += 1
    
    left_component : list[Point] = line_coordinates[:insert_index]
    left_component.append(point)
    right_component : list[Point] = [point]
    right_component.extend(line_coordinates[insert_index:])
    
    return (LineString(left_component),LineString(right_component))
    

def generate_straight_segments_recursive(features_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Recursively process the passed in arguments and return the gdf with the features that contain stragith geometries.
    
    ### Parameters
        features_gdf: **geopandas.GeoDataFrame** - Dataframe to be used for validation of geometries
        
    ### Returns
        A **geopandas.GeoDataFrame** object with the geometry column containing straight segments from the original geometries.
    
    """
    
    # Step 1: Generate straight distances for each geometry and check the percent difference between the actual length and straight distance length
    features_gdf = features_gdf.copy()
    features_gdf['Straight_Distances'] = features_gdf.apply(lambda x: LineString([x['geometry'].coords[0],x['geometry'].coords[-1]]).length,axis=1)
    features_gdf['Percent_differences'] = abs(features_gdf.geometry.length -  features_gdf['Straight_Distances']) / features_gdf['Straight_Distances']
    
    # Step 2: Create three subsets (geometries with straight segments, geometries that will contain right split, and geometries that will contain left splits)
    threshold = 0.005
    straight_segments = features_gdf[features_gdf['Percent_differences'] <= threshold].copy().reset_index(drop=True)
    non_straight_segments_split_one = features_gdf[features_gdf['Percent_differences'] > threshold].copy().reset_index(drop=True)
    non_straight_segments_split_two = features_gdf[features_gdf['Percent_differences'] > threshold].copy().reset_index(drop=True)
    
    if straight_segments.shape[0] != features_gdf.shape[0]:
        
        # Step 3a: Populate split subsets with geometries (If any non-straight geometries)
        non_straight_segments_split_one['midpoints'] = non_straight_segments_split_one.apply(lambda x: line_interpolate_point(x['geometry'],x['geometry'].length / 2),axis=1)
        non_straight_segments_split_one['geometry'] = non_straight_segments_split_one.apply(lambda x: generate_split(x['geometry'],x['midpoints'])[0],axis=1)
        
        non_straight_segments_split_two['midpoints'] = non_straight_segments_split_two.apply(lambda x: line_interpolate_point(x['geometry'],x['geometry'].length / 2),axis=1)
        non_straight_segments_split_two['geometry'] = non_straight_segments_split_two.apply(lambda x: generate_split(x['geometry'],x['midpoints'])[1],axis=1)

    
        # Step 4: Recurse on splits
        results_left_split = generate_straight_segments_recursive(non_straight_segments_split_one)
        results_right_split = generate_straight_segments_recursive(non_straight_segments_split_two)
        
        # Step 5: Compare the two splits and keep the ones that contain the longer geometry
        
        # Sanity checks - Make sure that the splits' sizes and ordering remained the same
        assert results_left_split.shape[0] == results_right_split.shape[0], "Splits not the same size"
        assert results_left_split[results_left_split['Estimation'].isin(results_right_split['Estimation'].tolist())].shape[0] == results_left_split.shape[0], "Inconsistent IDs"
        
        geometries = []
        estimate_ids = results_left_split['Estimation'].tolist()
        
        for estimation_id in estimate_ids:
            line_left : LineString = results_left_split['geometry'][results_left_split['Estimation'] == estimation_id].tolist()[0]
            line_right : LineString = results_right_split['geometry'][results_left_split['Estimation'] == estimation_id].tolist()[0]
            assert isinstance(line_left,LineString) == True, 'Line from left split is not of type shapely.LineString'
            assert isinstance(line_right,LineString) == True, 'Line from right split is not of type shapely.LineString'
            
            if line_left.length > line_right.length:
                geometries.append(line_left)
            else:
                geometries.append(line_right)
                
        results_left_split['geometry'] = gpd.GeoSeries(geometries)
        
        # Step 6: Return straight segments alonside remidiated segments
        results_left_split = results_left_split[straight_segments.columns]
        return_result_df = pd.concat([results_left_split,straight_segments],axis=0,ignore_index=True)
        return return_result_df
        
        
    else:
        # Step 3b: Return straight segments if no more geometries contain curved segments
        return straight_segments
    
    

def generate_straight_segments(path:str) -> gpd.GeoDataFrame:
    """
    Given the path to shp file, generate straight vector geometries for each feature.
    
    ### Parameters
        path: **str** - Path to the shp file
    
    ### Returns
        A **geopandas.GeoDataFrame** object with the geometry column containing straight segments from the original geometries.
    """
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=3776)
    
    # Convert the geometry from LINESTRING Z to LINESTRING
    gdf['geometry'] = gdf.apply(lambda x: force_2d(x['geometry']),axis=1)
    straight_segments_gdf = generate_straight_segments_recursive(gdf)
    straight_segments_gdf['Straight_Distances'] = straight_segments_gdf.apply(lambda x: LineString([x['geometry'].coords[0],x['geometry'].coords[-1]]).length,axis=1)
    straight_segments_gdf['Percent_differences'] = abs(straight_segments_gdf.geometry.length -  straight_segments_gdf['Straight_Distances']) / straight_segments_gdf['Straight_Distances']
    
    print(straight_segments_gdf.shape)
    print(straight_segments_gdf[straight_segments_gdf['Percent_differences'] > 0.005].shape[0])
    
    
    

if __name__ == "__main__":
    shp_file_path = r"C:\Users\abhin\OneDrive\Documents\ArcGIS\Projects\Research\Road Network\Set2-Roads.shp"
    gdf = generate_straight_segments(shp_file_path)