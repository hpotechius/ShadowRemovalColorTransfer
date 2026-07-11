"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import bpy
import os
import json
from mathutils import Vector
from datetime import datetime

##############################################################################
# Function to create sunlight
############################################################################## 
def create_sunlight():
    # Create a new light datablock
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    
    # Create a new object with the light datablock
    light_object = bpy.data.objects.new(name="Sun", object_data=light_data)
    
    # Link the light object to the collection
    bpy.context.collection.objects.link(light_object)
    
    return light_object

##############################################################################
# Function to set the sun position using the Sun Position addon
############################################################################## 
def set_sun_position(sun_object, dateloc):
    # Ensure the sun object is the active object
    bpy.context.view_layer.objects.active = sun_object
    
    # Set the location of the sun object (optional, for better visualization)
    sun_object.location = (0, 0, 10)
    
    # Use the Sun Position addon to set the sun parameters
    sun_position = bpy.context.scene.sun_pos_properties
    sun_position.use_sun = True
    sun_position.sun_object = sun_object
    
    # Set the date and time for the sun position
    date = datetime.now().date()
    time = datetime.strptime(dateloc["time"], "%H:%M").time()
    
    sun_position.year = dateloc["year"]
    sun_position.month = dateloc["month"]
    sun_position.day = dateloc["day"]
    sun_position.UTC_zone = dateloc["utc"]
    sun_position.latitude = dateloc["latitude"]
    sun_position.longitude = dateloc["longitude"]
    sun_position.time = time.hour + time.minute / 60
    
    sun_position.north_offset = dateloc["northoffset"]
    sun_position.use_daylight_savings = dateloc["daylightsavings"]
    
##############################################################################
# Function to add an image texture node to each material of the object
##############################################################################    
def add_image_texture_to_materials(obj, image):
    for material_slot in obj.material_slots:
        mat = material_slot.material
        if mat:
            # Ensure the material uses nodes
            mat.use_nodes = True
            nodes = mat.node_tree.nodes

            # Create an image texture node
            image_texture_node = nodes.new(type='ShaderNodeTexImage')
            image_texture_node.location = (-400, 400)
            
            # Assign the existing image to the texture node
            image_texture_node.image = image
            
            # Select the image texture node
            for node in nodes:
                node.select = False
            image_texture_node.select = True
            mat.node_tree.nodes.active = image_texture_node
            
            print(f"Added image texture node to material '{mat.name}' of object '{obj.name}'")
        
##############################################################################
# Function to bake the diffuse color to the image
############################################################################## 
def bake_diffuse_color(obj):
    # Set bake type to 'DIFFUSE'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'

    # Set the contributions to only 'COLOR'
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True

    # Set the margin size
    bpy.context.scene.render.bake.margin = 5
    bpy.context.scene.render.bake.margin_type = 'EXTEND'

    # Bake the diffuse color
    bpy.ops.object.bake(type='DIFFUSE')
    
##############################################################################
# Function to bake the diffuse color to the image
############################################################################## 
def bake_shading(obj):
    # Set bake type to 'DIFFUSE'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'

    # Set the contributions to only 'COLOR'
    bpy.context.scene.render.bake.use_pass_direct = True
    bpy.context.scene.render.bake.use_pass_indirect = True
    bpy.context.scene.render.bake.use_pass_color = True

    # Set the margin size
    bpy.context.scene.render.bake.margin = 5
    bpy.context.scene.render.bake.margin_type = 'EXTEND'

    # Bake the diffuse color
    bpy.ops.object.bake(type='DIFFUSE')
      
##############################################################################
# Function to bake the diffuse color to the image
##############################################################################
def bake_normal(obj):
    # Set bake type to 'DIFFUSE'
    bpy.context.scene.cycles.bake_type = 'NORMAL'

    # Set the contributions to only 'COLOR'
    bpy.context.scene.render.bake.normal_space = "OBJECT"

    # Set the margin size
    bpy.context.scene.render.bake.margin = 5
    bpy.context.scene.render.bake.margin_type = 'EXTEND'
    
    # Set the swizzle parameters
    bpy.context.scene.render.bake.normal_r = 'POS_X'  # X-normals to Red
    bpy.context.scene.render.bake.normal_g = 'NEG_Z'  # Y-normals to Green
    bpy.context.scene.render.bake.normal_b = 'POS_Y'  # Z-normals to Blue, or 'POS_X' if you 

    # Bake the diffuse color
    bpy.ops.object.bake(type='NORMAL')
    
##############################################################################
# Function to bake the diffuse color to the image
############################################################################## 
def bake_diffuse_height(obj):
    # Set bake type to 'DIFFUSE'
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'

    # Set the contributions to only 'COLOR'
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True

    # Set the margin size
    bpy.context.scene.render.bake.margin = 5
    bpy.context.scene.render.bake.margin_type = 'EXTEND'

    # Bake the diffuse color
    bpy.ops.object.bake(type='DIFFUSE')
    
##############################################################################
# Function to create a new material with the image "SingleTex" and assign it to the object
############################################################################## 
def create_and_assign_singletex_material(obj, image_name="SingleTex", uv_map_name="UVMap_new"):
    # Create a new material
    material = bpy.data.materials.new(name=image_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Create an image texture node and assign the image
    image_texture_node = nodes.new(type='ShaderNodeTexImage')
    image_texture_node.image = bpy.data.images[image_name]

    # Connect the image texture node to the material output
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        material.node_tree.links.new(image_texture_node.outputs['Color'], bsdf.inputs['Base Color'])

    # Delete all other materials on the object
    while obj.material_slots:
        obj.active_material_index = 0
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.material_slot_remove()
    
    # Assign the new material to the object
    obj.data.materials.append(material)
    
    # Activate the UV map "UVMap_new" for rendering
    if uv_map_name in obj.data.uv_layers:
        obj.data.uv_layers.active = obj.data.uv_layers[uv_map_name]
        for uv_map in obj.data.uv_layers:
            if uv_map.name == uv_map_name:
                uv_map.active_render = True
                print(f"Activated UV map '{uv_map_name}' for rendering on object '{obj.name}'")
    else:
        print(f"UV map '{uv_map_name}' not found on object '{obj.name}'")

    print(f"Created and assigned material '{image_name}' to object '{obj.name}'")
        
##############################################################################
# 
##############################################################################         
def create_and_assign_white_material(obj, material_name="WhiteMaterial"):
    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Clear all nodes to start fresh
    for node in nodes:
        nodes.remove(node)

    # Add a Principled BSDF shader node
    bsdf = nodes.new(type='ShaderNodeBsdfDiffuse')
    bsdf.location = (0, 0)
    bsdf.inputs['Color'].default_value = (1, 1, 1, 1)  # Set the color to white

    # Add a Material Output node
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (200, 0)

    # Connect the BSDF shader to the Material Output
    material.node_tree.links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Delete all other materials on the object
    while obj.material_slots:
        obj.active_material_index = 0
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.material_slot_remove()

    # Assign the new material to the object
    obj.data.materials.append(material)

    print(f"Created and assigned white material '{material_name}' to object '{obj.name}'")

##############################################################################
# 
##############################################################################  
def create_and_assign_gradient_material(obj, material_name="GradientMaterial"):
    # Create a new material
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Clear all nodes to start fresh
    for node in nodes:
        nodes.remove(node)

    # Add nodes
    texture_coord = nodes.new(type='ShaderNodeTexCoord')
    texture_coord.location = (-400, 0)

    separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = (-200, 0)
    
    bounding_box_min_y = min((Vector(corner)).y for corner in obj.bound_box)
    bounding_box_max_y = max((Vector(corner)).y for corner in obj.bound_box)

    # Calculate bounding box min and max Y coordinates
    y_range = bounding_box_max_y - bounding_box_min_y

    # Add Map Range node to normalize Y coordinates
    map_range = nodes.new(type='ShaderNodeMapRange')
    map_range.location = (0, 0)
    map_range.inputs['From Min'].default_value = bounding_box_min_y
    map_range.inputs['From Max'].default_value = bounding_box_max_y
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0
    map_range.interpolation_type = 'SMOOTHERSTEP'  # Set interpolation to smoother step

    # Add color ramp for the gradient
    gradient = nodes.new(type='ShaderNodeValToRGB')
    gradient.location = (200, 0)
    gradient.color_ramp.interpolation = 'LINEAR'
    gradient.color_ramp.elements[0].position = 0.0
    gradient.color_ramp.elements[0].color = (0, 0, 0, 1)
    gradient.color_ramp.elements[1].position = 1.0
    gradient.color_ramp.elements[1].color = (1, 1, 1, 1)
    
    # Create a Diffuse BSDF node
    diffuse_bsdf = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse_bsdf.location = (400, 0)

    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    material_output.location = (600, 0)

    # Link nodes
    material.node_tree.links.new(texture_coord.outputs['Object'], separate_xyz.inputs['Vector'])
    material.node_tree.links.new(separate_xyz.outputs['Y'], map_range.inputs['Value'])
    material.node_tree.links.new(map_range.outputs['Result'], gradient.inputs['Fac'])
    material.node_tree.links.new(gradient.outputs['Color'], diffuse_bsdf.inputs['Color'])
    material.node_tree.links.new(diffuse_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    # Delete all other materials on the object
    while obj.material_slots:
        obj.active_material_index = 0
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.material_slot_remove()

    # Assign the new material to the object
    obj.data.materials.append(material)

    print(f"Created and assigned gradient material '{material_name}' to object '{obj.name}'")
    
##############################################################################
# Function to export the object as OBJ with PNG texture 
##############################################################################  
def export_obj_with_png(obj, export_path, texture_path, img_name):
    # Select the object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Save the image as PNG
    image = bpy.data.images[img_name]
    image.filepath_raw = texture_path
    image.file_format = 'PNG'
    image.save()

    # Export the object as OBJ
    bpy.ops.wm.obj_export(filepath=export_path)

    # Deselect the object
    obj.select_set(False)

    print(f"Exported '{obj.name}' as OBJ to '{export_path}' with texture '{texture_path}'")
            
##############################################################################
# Function to clear the current scene
##############################################################################
def clear_scene():
    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    # Delete all selected objects
    bpy.ops.object.delete()
    
    # Remove all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Remove all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)

##############################################################################
# Read files from given folder
##############################################################################
# Load options from JSON file
with open('options.json', 'r') as f:
    options = json.load(f)

directory = options['in_folder']
export_directory = options['in_folder']
dateloc = options['dateloc']

bpy.context.scene.render.engine = 'CYCLES'

# Get a list of all files in the directory
files = os.listdir(directory)
# Filter out files to get only .obj files (or any other specific file format)
obj_files = [f for f in files if f.endswith('.obj')]

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Loop through all .obj files and import them into Blender
for obj_file in obj_files:
    #type_export = ["reflectance", "shading", "height", "normal","texture","mask"]
    type_export = ["shading", "height", "normal", "mask"]

    for type_e in type_export:
        print(f"START: {type_e}")
        clear_scene()

        # Create a single image to be used by all materials
        image_base_name = os.path.splitext(obj_file)[0]
        
        if type_e == "texture" or type_e == "shading":
            image_name = f"{image_base_name}_{type_e}"
            export_path = os.path.join(export_directory, f"{os.path.splitext(obj_file)[0]}_{type_e}.obj")
            print(f"Export path for {type_e}: {export_path}")
        else:
            image_name = f"{image_base_name}_{type_e}"
            export_path = os.path.join(export_directory, f"{os.path.splitext(obj_file)[0]}_{type_e}.obj")
            print(f"Export path for {type_e}: {export_path}")

        clear_scene()
            
        # Define the export paths)
        texture_path = os.path.join(export_directory, f"{image_name}.png")
            
        image_width = 1024
        image_height = 1024
        image = bpy.data.images.new(name=image_name, width=image_width, height=image_height)
        print(f"Created image '{image_name}' with dimensions {image_width}x{image_height}")

        # Ensure directory for the current image_name
        ensure_directory(export_directory)

        file_path = os.path.join(directory, obj_file)
        
        # Import the .obj file
        bpy.ops.wm.obj_import(filepath=file_path)
        
        # Get the imported object (assume it's the only object in the scene)
        imported_obj = bpy.context.selected_objects[0]
        
        # Bake the diffuse color to the image
        if type_e == "reflectance":
            # Add an image texture node to each material of the object
            add_image_texture_to_materials(imported_obj, image)
            bake_diffuse_color(imported_obj)
        elif type_e == "normal":
            # Add an image texture node to each material of the object
            add_image_texture_to_materials(imported_obj, image)
            bake_normal(imported_obj)
        elif type_e == "mask":
            create_and_assign_white_material(imported_obj)
            # Add an image texture node to each material of the object
            add_image_texture_to_materials(imported_obj, image)
            bake_diffuse_color(imported_obj)
        elif type_e == "height":
            create_and_assign_gradient_material(imported_obj)
            # Add an image texture node to each material of the object
            add_image_texture_to_materials(imported_obj, image)
            bake_diffuse_height(imported_obj)
        elif type_e == "texture":
            # Create sunlight
            sunlight = create_sunlight()
            # Set the sun position to 16:45
            set_sun_position(sunlight, dateloc)
            # Ensure the object is active before calling the function
            bpy.context.view_layer.objects.active = imported_obj
            
            # Add an image texture node to each material of the object
            add_image_texture_to_materials(imported_obj, image)
            bake_shading(imported_obj)
        elif type_e == "shading":
            # Create sunlight
            sunlight = create_sunlight()
            # Set the sun position to 16:45
            set_sun_position(sunlight, dateloc)
            # Ensure the object is active before calling the function
            bpy.context.view_layer.objects.active = imported_obj

            create_and_assign_white_material(imported_obj)
            # Add an image texture node to each material of the object
            add_image_texture_to_materials(imported_obj, image)
            bake_shading(imported_obj)
        
        # Create and assign the "SingleTex" material with the baked image
        create_and_assign_singletex_material(imported_obj, image_name)
        
        # Export
        export_obj_with_png(imported_obj, export_path, texture_path, image_name)
        
        print(f"Imported: {file_path}")

print("All files imported.")