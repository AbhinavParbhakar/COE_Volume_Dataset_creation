import geopandas as gpd
from shapely import LineString
import numpy as np
from shapely.ops import split, snap



def generate_straight_segments_recursive(features_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Recursively process the passed in arguments and return the gdf with the features that contain stragith geometries.
    
    ### Parameters
        features_gdf: **geopandas.GeoDataFrame** - Dataframe to be used for validation of geometries
        
    ### Returns
        A **geopandas.GeoDataFrame** object with the geometry column containing straight segments from the original geometries.
    
    """
    features_gdf = features_gdf.copy()
    features_gdf['Straight_Distances'] = features_gdf.apply(lambda x: LineString([x['geometry'].coords[0],x['geometry'].coords[-1]]).length,axis=1)
    features_gdf['Percent_differences'] = abs(features_gdf.geometry.length -  features_gdf['Straight_Distances']) / features_gdf['Straight_Distances']
    straight_segments = features_gdf[features_gdf['Percent_differences'] == 0.0].copy().reset_index()
    non_straight_segments_split_one = features_gdf[features_gdf['Percent_differences'] != 0.0].copy().reset_index()
    non_straight_segments_split_two = features_gdf[features_gdf['Percent_differences'] != 0.0].copy().reset_index()
    
    non_straight_segments_split_one['geometry'] = non_straight_segments_split_one.apply(lambda x: split(snap(x['geometry'],x['geometry'].centroid,tolerance=0.49),x['geometry'].centroid).geoms.__len__(),axis=1)
    # non_straight_segments_split_two['geometry'] = non_straight_segments_split_two.apply(lambda x: split(x['geometry'],x['geometry'].centroid).geoms[1],axis=1)
    
    print(non_straight_segments_split_one['geometry'][:2])
    
    
    
    
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
    generate_straight_segments_recursive(gdf)
    
    

if __name__ == "__main__":
    shp_file_path = r"C:\Users\abhin\OneDrive\Documents\ArcGIS\Projects\Research\Road Network\Set2-Roads.shp"
    gdf = generate_straight_segments(shp_file_path)