import ee
import geemap
from shapely.wkt import loads
from shapely import LineString, to_geojson
import pandas as pd
import ast
from playwright.sync_api import sync_playwright
import os
import time


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
    graph = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(line)
        .filterDate('2019-01-01', '2019-12-31')
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
        time.sleep(5)
        page.screenshot(path=save_name,full_page=True,clip=screenshot_settings)
    
    if delete_html_file:
        os.remove(file_name)
    

def create_granular_output(line:ee.geometry.Geometry,file_name:str):
    """
    Given the line and the file name, created images with the Estimation Id as the name and 
    stores it in images folder
    """
    html_file = create_output_html(line,file_name)
    estimation_point_id = os.path.basename(html_file).split('.')[0]
    save_name = f'./data/images/{estimation_point_id}.png'
    create_zoomed_in_images(html_file,save_name=save_name,delete_html_file=True)

def create_granular_images(data_file)->list[str]:
    """
    Given the data file containing geometries, see what one looks like
    on the screen.
    """
        
    ee.Authenticate()
    ee.Initialize(project='ee-city-of-edmonton-modeling')
    df = pd.read_csv(data_file)
    df['ee_geometry'] = df.apply(lambda x: shapely_to_ee_geom(loads(x['road_geo'])),axis=1)
    file_names = df.apply(lambda x: create_granular_output(x['ee_geometry'],f'./data/html_files/{x['Estimation_point']}.html'),axis=1)
    
    return file_names
    

if __name__ == "__main__":
    data_file = './data/excel_files/data_with_lines.csv'
    create_granular_images(data_file)
        
        