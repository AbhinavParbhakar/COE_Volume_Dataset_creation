import geopandas as gpd
from shapely.geometry import Point, MultiPolygon
from shapely.wkt import loads
from pyproj import Transformer
import pandas as pd
import datetime

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

def intersection_to_midblock(df:pd.DataFrame,directions:list,aux_cols:list,offset=20)->pd.DataFrame:
    """
    Input a df and redefine all intersections as midblocks with seperated volumes. This is done by calculating each point in EPSG:4236, 
    and then projecting to EPSG:32612, the UTM zone for Edmonton, adding X number for midblocks for each intersection, and then adding X new points
    with new coordinates as defined as x meters away from each center with the corresponding volume for that direction. Offset distance (m) is applied 
    when making the transformation.
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
                        y += offset
                    case "N Volume":
                        y -= offset
                    case "E Volume":
                        x -= offset
                    case "W Volume":
                        x += offset
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

def pair_road_class(path:str,df:pd.DataFrame,desired_cols=['Lat','Long','roadclass','Volume'])->pd.DataFrame:
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
    road_class_gf = road_class_gf.to_crs(epsg=utm)
    
    # Step 4
    merged_gf = features_gf.sjoin_nearest(right=road_class_gf,how='left',distance_col='distance')
    merged_gf = merged_gf[~merged_gf.index.duplicated(keep='first')]
        
    # Step 5
    return merged_gf[desired_cols]

def pair_unique_id(df:pd.DataFrame,cols=['Lat','Long'])->pd.DataFrame:
    """
    Given a df, and cols, create a new column in the returned dataframe based
    off a hash of the values of the listed columns combined.
    
    Returns: DataFrame
    """
    df = df.copy(deep=True)
    df['UniqueID'] = df.apply(generate_hash,args=[cols],axis=1)
    
    assert df.shape[0] == len(df['UniqueID'].unique()), "Id's generated are not unique"
    return df

def generate_hash(row:dict,cols:list)->str:
    """
    Generate a hash based on the concatenated columns provided
    
    Return the hash
    """
    input_val = ''.join([str(row[col]) for col in cols])
    return hash(input_val)

def pair_speed(df:pd.DataFrame,shp_file:str,desired_cols=['UniqueID','Lat','Long','roadclass','Volume','Speed (km/h)'])->pd.DataFrame:
    """
    Given the df, the shp_file, and desired_cols, match the speed of the closest road from the shape_file, and return a DataFrame with the given cols.
    
    1. Turn the longitude and latitude points of each sample in shapely.geometry.Point, and
    turn the df into a GeoDataFrame, and project this into UTM-12
    2. Map the GeoDataFrame from the shp_file into UTM-12
    3. Combine the two, and delete any duplicates
    4. Return the joined data with the desired columns
    """
    geo = 4326
    utm = 32612
    
    # Step 1
    df = df.copy(deep=True)
    df['Geometry'] = df.apply(lambda x: Point(x['Long'],x['Lat']),axis=1)
    excel_gdf = gpd.GeoDataFrame(data=df,geometry='Geometry',crs=f'EPSG:{geo}')
    excel_gdf = excel_gdf.to_crs(epsg=utm)
    
    # Step 2
    speed_gdf : gpd.GeoDataFrame = gpd.read_file(shp_file)
    speed_gdf = speed_gdf.to_crs(epsg=utm)
    speed_gdf['speed_int'] = speed_gdf['speed'].astype(int)
    speed_gdf['saved_geo'] = speed_gdf['geometry']
    
    
    # Step 3
    joined_data = excel_gdf.sjoin_nearest(right=speed_gdf,how='left',distance_col='distance')
    duplicate_index = joined_data.index.duplicated(keep=False)
    
    # Impute the mean speed between duplicates. All duplicates had only two matches, 30 and 40km/h, so this is not a bad approach
    duplicate_mean_speed = joined_data[duplicate_index].groupby(by='UniqueID',as_index=False)[['speed_int']].mean()
    joined_data.loc[duplicate_index,'speed_int'] = duplicate_mean_speed['speed_int']
    joined_data = joined_data[~joined_data.index.duplicated(keep='first')]
    
    
    print(joined_data.columns)
    # Step 4
    joined_data = joined_data.rename(mapper={'speed_int':'Speed (km/h)'},axis=1)
    
    print(joined_data.columns)
    
    return joined_data[desired_cols]
    
def pair_land_use(df:pd.DataFrame,exl_file:str,desired_cols=['UniqueID','Lat','Long','roadclass','Speed (km/h)','Land Usage','Volume'])->pd.DataFrame:
    """
    Given the dataframe, and the desired_cols, return a new DataFrame with land usage attached
    
    1. Create the points from the df,
    2. Export the excel file with land usage and turn it into a geoframe
    3. Join the two after projecting to utm
    4. Return the data
    """
    geo = 4326
    utm = 32612
    
    # Step 1
    df['Geometry'] = df.apply(lambda x: Point(x['Long'],x['Lat']),axis=1)
    excel_gdf = gpd.GeoDataFrame(data=df,geometry='Geometry',crs=f'EPSG:{geo}')
    excel_gdf = excel_gdf.to_crs(epsg=utm)
    
    # Step 2
    land_df = pd.read_excel(io=exl_file)
    land_df['Geometry'] = land_df.apply(func=retrieve_multipolygon,axis=1,args=('geometry_multipolygon'))

def retrieve_multipolygon(row:dict,col_name:str)-> MultiPolygon:
    """
    Given the row, and the col_name, turn thes string value in the corresponding
    key based on the col_name, and turn in into a MultiPolygon
    """
    return loads(row[col_name])

if __name__ == "__main__":
    # Do not delete
    now = datetime.datetime.now()
    df_save_name = f'./data/excel_files/features{now.month}-{now.day}-{now.year}.xlsx'
    roadclass_shp_file = './data/shape_files/RoadClass_CoE.shp'
    speed_shp_file = './data/shape_files/Speed limit Shapfile.shp'
    land_usage_file = './data/excel_files/Land Use Features.xlsx'
    
    # Start from here
    df = pd.read_excel(df_save_name)
    pair_land_use(df,land_usage_file)
    
    
