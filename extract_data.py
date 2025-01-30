import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
import pandas as pd

def clean_adt_volume(path:str)->pd.DataFrame:
    """
    Given a path to an excel file containing volume data, a new DataFrame will be created and returned that:
    
    1. Removes all rows that contain Pedestrian Volume, and sub NA's as 0
    2. Splits the studies into midblocks and intersections.
    3. Calculates the adj. total volume for midblocks
    4. Calculate the adj. Direction in volume for all directions existing in an intersection, and add them up to create a new total vol for each direction
    5. Calculate the mean volume for each distinct study within midblocks
    6. Calculate the mean volume for each direction for each study within intersections
    7. Redefine all intersections as midblocks with seperated volumes. This is done by calculating each point in EPSG:4236, 
    and then projecting to EPSG:32612, the UTM zone for Edmonton, adding X number for midblocks for each intersection, and then adding X new points
    with new coordinates as defined as 8 meters away from each center with the corresponding volume for that direction
    8. Return the concatentated results from midblocks and intersections
    """

    df = pd.read_excel(path)

    # Step 1:
    df = df.drop(df.index[df['Study Type'].str.contains('Ped ')])
    df = df.fillna(0)
    
    # Step 2:
    midblocks = df[df['Road Segment Type'] == 'Midblock']
    intersections = df[df['Road Segment Type'] == 'Intersection']

    # Step 3: 
    midblocks = midblocks.reset_index(drop=True)
    inputs = ["Lights", "Buses", "Articulated Trucks", "Motorcycles", "Cars", "Light Goods Vehicles", "Single-Unit Trucks", "Articulated Trucks", "Heavy", "Heavy and Lights"]
    midblocks['Volume'] = pd.Series([0 for i in range(midblocks.shape[0])])
    for input in inputs:
        midblocks['Volume'] += midblocks[input]
    
    # Step 4:
    intersections = intersections.reset_index(drop=True)
    directions = ['S','N','E','W']
    for direction in directions:
        intersections[f'{direction} Volume In'] = pd.Series([0 for i in range(intersections.shape[0])])
        intersections[f'{direction} Volume'] = pd.Series([0 for i in range(intersections.shape[0])])
    
    for direction in directions:
        for input in inputs:
            intersections[f'{direction} Volume In'] += intersections[f'{direction} {input}']
        intersections[f'{direction} Volume'] = intersections[f'{direction} Volume In'] + intersections[f'{direction} Adj. Out']
    
    # Step 5:
    aux_cols = ['Id','Lat','Long','Location','Road Segment Type','Date']
    mean_cols = ['Volume']
    distinct_midblocks = aggregate_means(midblocks,mean_cols,aux_cols,'Location')
    
    # Step 6:
    mean_cols = ['S Volume', 'N Volume', 'E Volume', 'W Volume']
    distinct_intersections = aggregate_means(intersections,mean_cols,aux_cols,'Location')
    
    # Step 7:
    new_midblocks = intersection_to_midblock(distinct_intersections,mean_cols,aux_cols)
    
    # Step 8:
    final_df = pd.concat([distinct_midblocks,new_midblocks])
    final_df['Volume'] = final_df['Volume'].astype(int)
    final_df = final_df.drop(columns=['Road Segment Type','Location'])
    return final_df.reset_index(drop=True)

def aggregate_means(df:pd.DataFrame,mean_cols:list,aux_cols:list,group_by:str)->pd.DataFrame:
    """
    Given a DataFrame, group by the passed in parameter and calculate the mean over the provided cols
    and store them in new cols for each column. Additionally, add the provide auxillary information.
    
    Return the resulting DataFrame
    """
    seen_value = {}

    new_df_dict : dict[str,list] = {}
    
    for col in mean_cols:
        new_df_dict[col] = []
    for col in aux_cols:
        new_df_dict[col] = []
        
    for value, group in iter(df.groupby(by=group_by)):
        if value not in seen_value:
            seen_value[value] = True
            group = group.reset_index(drop=True)
            
            for col in mean_cols:
                mean = group[col].mean()
                new_df_dict[col].append(mean)
            
            for col in aux_cols:
                new_df_dict[col].append(group.loc[0,col])
    
    return pd.DataFrame(data=new_df_dict)

def intersection_to_midblock(df:pd.DataFrame,directions:list,aux_cols:list)->pd.DataFrame:
    """
    Input a df and redefine all intersections as midblocks with seperated volumes. This is done by calculating each point in EPSG:4236, 
    and then projecting to EPSG:32612, the UTM zone for Edmonton, adding X number for midblocks for each intersection, and then adding X new points
    with new coordinates as defined as 8 meters away from each center with the corresponding volume for that direction. 
    """
    df = df.reset_index(drop=True)
    to_utm = Transformer.from_crs("EPSG:4326","EPSG:32612",always_xy=True)
    to_geo = Transformer.from_crs("EPSG:32612","EPSG:4326",always_xy=True)
    
    new_df_dict = {}
    new_df_dict['Volume'] = []
    
    for col in aux_cols:
        new_df_dict[col] = []
    
    for i in range(df.shape[0]):
        for dir in directions:
            if df.loc[i,dir] > 0.0:                
                lat = df.loc[i,'Lat']
                long = df.loc[i,'Long']
                
                x, y = to_utm.transform(long,lat)
                
                match dir:
                    case "S Volume":
                        y += 8
                    case "N Volume":
                        y -= 8
                    case "E Volume":
                        x -= 8
                    case "W Volume":
                        x += 8
                
                new_long,new_lat = to_geo.transform(x,y)
                
                df.loc[i,'Lat'] = new_lat
                df.loc[i,'long'] = new_long
                df.loc[i,'Road Segment Type'] = "Midblock"
                new_df_dict['Volume'].append(df.loc[i,dir])
                for col in aux_cols:
                    new_df_dict[col].append(df.loc[i,col])
                df.loc[i,'Lat'] = lat
                df.loc[i,'long'] = long
    
    return pd.DataFrame(data=new_df_dict)

def pair_road_class(path:str,df:pd.DataFrame)->pd.DataFrame:
    """
    Given a path to a .shp file, and a df containing cleaned information, class labels
    are generated based on the closness of fit from each point in the df to the lines in the .shp file.
    
    The following steps are taken:
    
    1. Lat and Long are required, and these columns are checked.
    2. These columns are used to create a geometry column with shapely.geometry.points, and fed into a GeoDataFrame with UTM-12.
    3. The shape file is read in as GeoDataFrame, and transformed into UTM-12 coordinates for comparison.
    4. The two GeoDataFrames are matched based on closeness of study points to lines.
    5. Final Resulting DataFrame is returned.
    
    A DataFrame containing all paired information is returned.
    """
    utm = 32612
    wsg = 4326
    
    # Step 1
    assert 'Lat' in df.columns,"Lat must be a column in the passed in DataFrame"
    assert 'Long' in df.columns, "Long must be a column in the passed in DataFrame"
    
    # Step 2
    df['Geometry'] = df.apply(lambda x: Point(x['Long'],x['Lat']),axis=1)
    features_gf = gpd.GeoDataFrame(data=df,geometry='Geometry',crs=f'EPSG:{wsg}')
    features_gf = features_gf.to_crs(epsg=utm)
    
    # Step 3
    road_class_gf : gpd.GeoDataFrame = gpd.read_file(path)
    road_class_gf = road_class_gf.to_crs(epsg=wsg)
    
    # Step 4
    merged_gf = features_gf.sjoin_nearest(right=road_class_gf,how='left',distance_col='distance')
    
    print(merged_gf[['Volume','Id','Lat','Long','roadclass','distance']][merged_gf.index.duplicated(keep=False)])
    print(merged_gf[merged_gf['distance'] == merged_gf['distance'].max()][['Lat','Long','roadclass']])
if __name__ == "__main__":
    aggregate_data_path = "./data/excel_files/Miovision Aggregate Data (Updated 2025).xlsx"
    output = clean_adt_volume(aggregate_data_path)
    output.to_excel('./data/excel_files/ADT_expanded_date_.xlsx',index=False)