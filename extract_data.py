import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
from pyproj import Transformer
import pandas as pd
import datetime
import matplotlib.pyplot as plt

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
    new_midblocks = intersection_to_midblock(distinct_intersections,mean_cols,aux_cols=['Id','Location','Road Segment Type','Date'])
    
    # Step 8:
    final_df = pd.concat([distinct_midblocks,new_midblocks])
    final_df['Volume'] = final_df['Volume'].astype(int)
    final_df = final_df.drop(columns=['Road Segment Type','Location'])
    return final_df.reset_index(drop=True)

def create_shape_file(df:pd.DataFrame,file_name="shape_file")->None:
    """
    Given a dataframe containing "Lat" and "Long" columns, create a shape file with the corresponding file name
    """
    lat_col_name = "Lat"
    long_col_name = "Long"
    lat_col_found = False
    long_col_found = False
    
    crs = 4326
    
    for x in df.columns.tolist():
        if x == lat_col_name:
            lat_col_found = True
        if x == long_col_name:
            long_col_found = True
    
    assert lat_col_found, "Lat column not present"
    assert long_col_found, "Long column no present"
    
    df = df.copy()
    df["Geometry"] = df.apply(lambda x: Point(x["Long"],x["Lat"]),axis=1)
    gf = gpd.GeoDataFrame(data=df,geometry="Geometry",crs=f"EPSG:{crs}")
    gf.to_file(f'./data/shape_files/{file_name}.shp')

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
    df = df.copy()
    df = df.reset_index(drop=True)
    
    to_utm = Transformer.from_crs("EPSG:4326","EPSG:32612",always_xy=True)
    to_geo = Transformer.from_crs("EPSG:32612","EPSG:4326",always_xy=True)
    
    new_df_dict = {}
    new_df_dict['Volume'] = []
    new_df_dict['Lat'] = []
    new_df_dict['Long'] = []
    
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
                

                df.loc[i,'Road Segment Type'] = "Midblock"
                new_df_dict['Volume'].append(df.loc[i,dir])
                new_df_dict['Lat'].append(new_lat)
                new_df_dict['Long'].append(new_long)
                for col in aux_cols:
                    new_df_dict[col].append(df.loc[i,col])

    
    new_midblocks = pd.DataFrame(data=new_df_dict)
    return new_midblocks

def pair_road_class(path:str,df:pd.DataFrame,desired_cols=['Lat','Long','roadclass','Volume','Date'])->pd.DataFrame:
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
    
    Return the hash as string value
    """
    input_val = ''.join([str(row[col]) for col in cols])
    return str(hash(input_val))

def pair_speed(df:pd.DataFrame,shp_file:str,desired_cols=['UniqueID','Lat','Long','roadclass','Volume','Speed (km/h)','Date'])->pd.DataFrame:
    """
    Given the df, the shp_file, and desired_cols, match the speed of the closest road from the shape_file, and return a DataFrame with the given cols.
    
    1. Turn the longitude and latitude points of each sample in shapely.geometry.Point, and
    turn the df into a GeoDataFrame, and project this into UTM-12
    2. Map the GeoDataFrame from the shp_file into UTM-12
    3. Combine the two, and combine the speed of duplicates grouped by UniqueID
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
    speed_gdf['speed'] = speed_gdf['speed'].astype(int)
    
    
    # Step 3
    joined_data = excel_gdf.sjoin_nearest(right=speed_gdf,how='left',distance_col='distance')
    duplicate_index = joined_data.index.duplicated(keep=False)
    
    # Impute the mean speed between duplicates. All duplicates had only two matches, 30 and 40km/h, so this is not a bad approach
    duplicate_mean_speed : pd.DataFrame = joined_data[duplicate_index].groupby(by='UniqueID',as_index=False)[['speed']].mean()
    speed_dict = dict(duplicate_mean_speed.apply(lambda x: (x['UniqueID'],x['speed']),axis=1).tolist())

    joined_data.loc[duplicate_index,'speed'] = joined_data.loc[duplicate_index,'UniqueID'].map(speed_dict)
    joined_data = joined_data[~joined_data.index.duplicated(keep='first')]
    
    
    # Step 4
    joined_data = joined_data.rename(mapper={'speed':'Speed (km/h)'},axis=1)
    return joined_data[desired_cols]
    
def pair_land_use(df:pd.DataFrame,file:str,desired_cols=['UniqueID','Lat','Long','roadclass','Speed (km/h)','Land Usage','Volume','Date'])->pd.DataFrame:
    """
    Given the dataframe, and the desired_cols, return a new DataFrame with land usage attached
    
    1. Create the points from the df,
    2. Export the csv file with land usage and turn it into a geoframe
    3. Join the two after projecting to utm, and remove any points that do not fall under
    any land usage zones, as well as removing any duplicates.
    4. Return the data
    """
    geo = 4326
    utm = 32612
    
    # Step 1
    df['Geometry'] = df.apply(lambda x: Point(x['Long'],x['Lat']),axis=1)
    excel_gdf = gpd.GeoDataFrame(data=df,geometry='Geometry',crs=f'EPSG:{geo}')
    excel_gdf = excel_gdf.to_crs(epsg=utm)
    
    # Step 2
    land_df = pd.read_csv(file)
    land_df['Geometry'] = land_df.apply(lambda x:loads(x['geometry_multipolygon']),axis=1)
    land_gdf = gpd.GeoDataFrame(data=land_df,geometry='Geometry',crs=f'EPSG:{geo}')
    land_gdf = land_gdf.to_crs(epsg=utm)
    land_gdf : gpd.GeoDataFrame = land_gdf[['Geometry','Category']]
        
    # Step 3
    joined_data : gpd.GeoDataFrame = excel_gdf.sjoin(df=land_gdf,how='left',predicate='within')
    joined_data = joined_data.drop(joined_data[joined_data['Category'].isna()].index,axis=0)
    
    # Based on manual check, keep last
    joined_data = joined_data[~joined_data.index.duplicated(keep='last')]
    joined_data = joined_data.rename(mapper={'Category':'Land Usage'},axis=1)
    
    # Step 4
    return_result = joined_data[desired_cols]
    return return_result

def missing_values(df:pd.DataFrame,verbose=False)->bool:
    """
    For a given df, checks each column to see if it is na, and returns true or false.
    
    Verbose option avaible to see name of column with missing data
    """
    cols = df.columns.to_list()
    
    i = 0
    missing = False
    
    while not missing and i < len(cols):
        length = len( df[cols[i]][df[cols[i]].isna()])
        
        if length > 0:
            missing = True
            if verbose:
                print('Missing column is',cols[i])
        i +=1
    
    return missing

def crop_buildings(file_path:str)->gpd.GeoDataFrame:
    """
    Given the file path for a .shp file detailing the buildings in all of Alberta, crop and clip based on the area
    around edmonton as specified by the polygon created.
    """
    coordinates = [[-113.335075,53.655685],
                [-113.690405,53.648387],
                [-113.687767,53.418376],
                [-113.320123,53.426762]]
    
    geo = 4326
    prs = 32612
    
    region_of_interest = Polygon(coordinates)
    map_info = {'name':['area'],'geometry':[region_of_interest]}
    roi = gpd.GeoDataFrame(data=map_info,geometry='geometry',crs=f'EPSG:{geo}')
    roi = roi.to_crs(epsg=prs)
    alberta_buildings : gpd.GeoDataFrame = gpd.read_file(file_path)
    alberta_buildings = alberta_buildings.to_crs(epsg=prs)
    
    cropped_area = alberta_buildings.clip(roi)
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    cropped_area.plot(ax=ax)
    ax.set_axis_off()
    plt.savefig('./data/graphs/COE_buildings.png')
    
    return cropped_area

if __name__ == "__main__":
    # Do not delete
    now = datetime.datetime.now()
    df_save_name = f'./data/excel_files/features{now.month}-{now.day}-{now.year}.xlsx'
    unclean_file = './data/excel_files/Miovision Aggregate Data (Updated 2025).xlsx'
    roadclass_shp_file = './data/shape_files/RoadClass_CoE.shp'
    speed_shp_file = './data/shape_files/Speed limit Shapfile.shp'
    land_usage_file = './data/excel_files/Land Use Features.csv'
    taz_file1 = './data/shape_files/TAZ1718.shp'
    buildings = './data/shape_files/gis_osm_buildings_a_free_1.shp'
    
    # Start from here
    df = clean_adt_volume(unclean_file)
    df = pair_road_class(roadclass_shp_file,df=df)
    df = pair_unique_id(df=df)
    df = pair_speed(df=df,shp_file=speed_shp_file)
    df = pair_land_use(df=df,file=land_usage_file)
    df.to_excel(df_save_name,index=False)
    
    
    
    
    
    
