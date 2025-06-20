import arcpy
import os

"C:/Users/abhin/OneDrive/Documents/ArcGIS/Projects/Research/Research.aprx"
# === USER CONFIGURATION ===
aprx_path = "C:/Users/abhin/OneDrive/Documents/ArcGIS/Projects/Research/Research.aprx"    # Path to your ArcGIS Pro project file
output_folder = "C:/Users/abhin/OneDrive/Documents/Computing/Research/City of Edmonton Volume Prediction/City of Edmonton Volume Data Creation/data/images"   # Folder to save exported images
vector_layer_name = "All_Points_Centroids"      # Exact vector layer name in the map
attribute_field = "Estimation"                        # Attribute field name to use for naming output files
buffer_distance = 100                           # Buffer distance around each feature extent (map units)

# === SCRIPT START ===

def main():
    # Open the ArcGIS Pro project
    aprx = arcpy.mp.ArcGISProject(aprx_path)

    # Get the first map (or specify by name if needed)
    # map_obj = aprx.listMaps()[0]

    # Get the first layout (or specify by name)
    layout = aprx.listLayouts()[1]
    print(f"The name of the layout is {layout.name}")

    # Get the map frame inside the layout
    map_frame = layout.listElements("MAPFRAME_ELEMENT")[0]
    print(map_frame.elementHeight,map_frame.elementWidth)
    map_frame.elementHeight,map_frame.elementWidth = 2.67,2.67
    print(map_frame.elementHeight,map_frame.elementWidth)
    map = map_frame.map
    camera = map_frame.camera
    camera_extent = camera.getExtent()
    camera_spatial_reference = camera_extent.spatialReference
    print(f"Camera extent is: \nBottom Left({camera_extent.XMin},{camera_extent.YMin}) \nUpper Right: ({camera_extent.XMax},{camera_extent.YMax})")
    print(f'Spatial Reference is {camera_extent.spatialReference.name}')
    layers = map.listLayers()
    for layer in layers:
        print(f'Layer: {layer.name}, Active: {layer.visible}')
    

    # # Get the vector layer by name
    vector_layer = map.listLayers(vector_layer_name)[0]
    vector_layer.visible = False
    road_layer = map.listLayers("All_Points")[0]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare fields list for cursor
    fields = [attribute_field, "SHAPE@"]
    i = 0
    with arcpy.da.SearchCursor(vector_layer, fields) as cursor:
        for row in cursor:
            i += 1
            attr_value = row[0]
            geom = row[1]
            # lanes = row[2]

            # Get geometry extent
            extent = geom.extent
            extent = extent.projectAs(camera_spatial_reference)
            # # Handle zero-area extent (e.g., points) by buffering
            # if extent.XMin == extent.XMax and extent.YMin == extent.YMax:
            #     # Point or zero-area geometry - create buffered extent around point
            # if lanes < 3:
            #     buffer = 20
            # elif lanes < 5:
            #     buffer = 30
            # else:
            buffer = 30
            buffered_extent = arcpy.Extent(
                extent.XMin - buffer,
                extent.YMin - buffer,
                extent.XMax + buffer,
                extent.YMax + buffer
            )
            # vector_layer.visible = False

            map_frame.camera.setExtent(buffered_extent)
            road_layer.visible = True
            query = f"ESTIMATION IN ({attr_value})"
            road_layer.definitionQuery = query

            safe_attr = str(attr_value).replace(" ", "_").replace("/", "_")

            out_path = os.path.join(output_folder, f"{int(float(safe_attr))}.png")
            
            
    
            pngFormat = arcpy.mp.CreateExportFormat('PNG',out_path)
            map_frame.export(pngFormat)
            print(f"Exported image for feature '{safe_attr}' -> {out_path}")

    print(f"All features processed and images exported ({i}).")

if __name__ == "__main__":
    main()
