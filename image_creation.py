import ee
import geemap
from shapely.wkt import loads
from shapely import LineString, to_geojson
import pandas as pd
import ast
from playwright.sync_api import sync_playwright
import os
import time
from multiprocessing import Pool
import multiprocessing
from bs4 import BeautifulSoup


def shapely_to_ee_geom(line:LineString)->ee.geometry.Geometry.LineString:
    """
    Given a shapely linestring, convert it to a 'ee' linestring
    """
    geo_json : dict = ast.literal_eval(to_geojson(line))
    return ee.geometry.Geometry.LineString(geo_json['coordinates'])

def create_output_html(line:ee.geometry.Geometry.LineString,file_name='output.html')->str:
    """
    Given the line string, creates an html file with an image
    zoomed in on that line by taking images
    from Sentinel-2. 
    """
    graph_centroid = ee.geometry.Geometry.Point(coords=[-113.206044,53.542114])
    graph = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(graph_centroid)
        .filterDate('2024-05-01', '2024-09-30')
        .sort('CLOUDY_PIXEL_PERCENTAGE')
        .first()
    )
    
    map = geemap.Map()
    map.center_object(line,15)
    map.add_layer(
        graph, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2000}, 'graph'
    )
    map.add_layer(
        line,
        {'color': 'FF0000','width':4},
        'Line'
    )
    
    map.save(file_name)
    
    return os.path.abspath(file_name)

def create_zoomed_in_images(file_name,save_name='Image.png',delete_html_file=False)->None:
    """
    Given a file containing a path to an html page, take a screenshot and then save that screenshot  
    with the given save name. The delete_html_file option can be passed in as true to delete the html file to free up space
    """
    
    # The settings treat the top left corner of the page as (0,0)
    screenshot_settings = {'height': 256, 'width': 256, 'x': 512, 'y': 163 }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(file_name)
        time.sleep(30)
        page.screenshot(path=save_name,full_page=True,clip=screenshot_settings)
    
    if delete_html_file:
        os.remove(file_name)
    

def create_granular_output(line:ee.geometry.Geometry,file_name:str):
    """
    Given the line and the file name, created images with the Estimation Id as the name and 
    stores it in images folder
    """
    ee.Authenticate()
    ee.Initialize(project='ee-city-of-edmonton-modeling')
    html_file = create_output_html(line,file_name)
    estimation_point_id = os.path.basename(html_file).split('.')[0]
    save_name = f'./data/images/{estimation_point_id}.png'
    create_zoomed_in_images(html_file,save_name=save_name,delete_html_file=True)

def create_granular_images(data_file,target_list=None)->list[str]:
    """
    Given the data file containing geometries, see what one looks like
    on the screen. Capable of only using the rows where the ID matches value in target
    """
        
    ee.Authenticate()
    ee.Initialize(project='ee-city-of-edmonton-modeling')
    df = pd.read_csv(data_file)
    
    if target_list:
        df = df[df.apply(lambda x: x['Estimation_point'] in target_list,axis=1)]
    
    df['ee_geometry'] = df.apply(lambda x: shapely_to_ee_geom(loads(x['road_geo'])),axis=1)
    arguments = df.apply(lambda x: (x['ee_geometry'],f'./data/html_files/{x['Estimation_point']}.html'),axis=1)
    
    with Pool(multiprocessing.cpu_count()) as p:
        p.starmap(create_granular_output,iterable=arguments.tolist())    
    return None


def create_coarse_output(excel_file:str,target_files=None)->None:
    """
    Given the excel file, map each location into a grid and generated that image.
    
    Parameters
    ----------
        excel_file : ``str``
            The path to the excel file containing the stored data.
        target_files : ``None`` or ``List[str]``
            The target id names (``None`` by default) for which to create the coarse images for.
    """
    ee.Authenticate()
    ee.Initialize(project='ee-city-of-edmonton-modeling')
    df = pd.read_csv(excel_file)
    df['ee_geometry'] = df.apply(lambda x: shapely_to_ee_geom(loads(x['road_geo'])),axis=1)
    arguments = df.apply(lambda x: (x['ee_geometry'],x['Estimation_point']),axis=1)
    
    with Pool(multiprocessing.cpu_count()) as p:
        p.starmap(create_coarse_images,iterable=arguments.tolist())


def capture_coarse_images(input_file:str,output_name:str,delete_html_file=False):
    # The settings treat the top left corner of the page as (0,0)
    screenshot_settings = {'height': 591, 'width': 709, 'x': 313, 'y': 16 }
    
    path = os.path.abspath(input_file)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_navigation_timeout(60000)
        page.goto(path,wait_until='domcontentloaded')
        time.sleep(30)
        page.screenshot(path=output_name,full_page=True,clip=screenshot_settings)
        
        page.close()
        browser.close()
    
    if delete_html_file:
        os.remove(input_file)
        
def create_coarse_html(road_segment:ee.geometry.Geometry,file_path:str):
    graph_centroid = ee.geometry.Geometry.Point(coords=[-113.206044,53.542114])
    zoom_centroid =   ee.geometry.Geometry.Point(coords=[-113.506804, 53.539434])
    bounding_box = ee.geometry.Geometry.BBox(north=53.658239,west=-113.730658,south= 53.399708,east=-113.244793)
    grids = geemap.fishnet(bounding_box,rows=15,cols=15)
    

    cropped_grids = grids.map(lambda f: f.set('keep',bounding_box.contains(f.geometry()))).filter(ee.filter.Filter.eq(name='keep',value=True))
    roi : ee.featurecollection.FeatureCollection = cropped_grids.map(lambda f: f.set('intersects',road_segment.intersects(f.geometry(),maxError=1))).filter(ee.filter.Filter.eq(name='intersects',value=True))

    grid_style =  {'color': '000000ff', 'width': 1, 'fillColor': '00000000'}
    bbox_style = {'color': 'ff0000ff', 'width': 2, 'fillColor': 'ff000080'}


    graph = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(graph_centroid)
    .filterDate('2019-05-01', '2019-09-30')
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .first()
    )
    
    map : geemap.Map = geemap.Map()
    map.center_object(zoom_centroid,11)
    
    map.add_layer(
        graph, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 2000}, 'graph'
    )
    map.add_layer(
        cropped_grids.style(**grid_style),
        {},
        'box'
    )
    
    map.add_layer(
        roi.style(**bbox_style),
        {},
        'Region of Interest'
    )
    
    map.save(file_path)

def clean_html_file(html_file_path:str):
    """
    Cleans the html file and removes unnecessary code that is not wanted and resaves the file.
    
    Parameters
    ----------
        html_file_path : str
            The path of the html file, for which the html tags should be removed
    """

    path = os.path.abspath(html_file_path)
    page_content = ""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(path)
        time.sleep(10)
        page_content = page.content()
    
    html = BeautifulSoup(page_content,'html.parser')
    for tag in html.find_all("div",attrs={'class':'leaflet-control-container'}):
        tag.decompose()
    
    with open(html_file_path,'w') as file:
        file.write(html.prettify())

def create_coarse_images(road_segment:ee.geometry.Geometry, file_name: str):
    """
    Create the image files containing the filled in coarse grids 
    Parameters
    ----------
        road_segment: ``ee.geometry.Geometry``
            The geometry of the selected road segment
        file_name : ``str``
            The name of the file under which the html should be saved
    """
    ee.Authenticate()
    ee.Initialize(project='ee-city-of-edmonton-modeling')
    html_save_path = f'./data/html_files/{file_name}.html'
    image_save_path = f'./data/images/{file_name}.png'
    create_coarse_html(road_segment=road_segment,file_path=html_save_path)
    clean_html_file(html_save_path)
    capture_coarse_images(input_file=html_save_path,output_name=image_save_path,delete_html_file=True)
    
    
    

if __name__ == "__main__":
    data_file = './data/excel_files/data_with_lines.csv'
    
    # files_to_delete = [102996, 104526, 514704, 514712, 515176, 530565, 530695, 530699, 530735, 530739, 530743, 966715, 967572, 1007611, 1017467, 1017468, 1095481, 1123698, 1121514, 1123614] 
    create_granular_images(data_file)
    
    #create_coarse_output(excel_file=data_file)
    
    # file_path = f'C:\\Users\\abhin\\OneDrive\\Documents\\Computing\\Research\\City of Edmonton Volume Prediction\\City of Edmonton Volume Data Creation\\data\\images'
   
    # files = [f'{file_path}\\{image}.png'  for image in files_to_delete]
    
    # for file in files:
    #     os.remove(file)
    # print('Done')    
        