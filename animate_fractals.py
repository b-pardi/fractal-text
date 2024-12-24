from manim import *

import json

with open("config.json", 'r') as f:
    anim_config = json.load(f)

# Configure Manim settings
config.frame_rate = anim_config['animation_framerate']
config.pixel_width = anim_config['animation_resolution_width']
config.pixel_height = anim_config['animation_resolution_height']
config.output_file = anim_config['animation_output_fname']
#config.background_color = config  # Optional: Set background color

def split_vgroup(vgroup, chunk_size):
    """Splits a VGroup into a list of smaller VGroups each containing chunk_size elements."""
    return [VGroup(*vgroup[i:i + chunk_size]) for i in range(0, len(vgroup), chunk_size)]

class FractalTextAnimation(Scene):
    def construct(self):
        global anim_config
        
        # Load fractal lines from JSON file
        with open("output/fractal_lines.json", "r") as f:
            fractal_lines = json.load(f)

        # Define scaling factors based on image dimensions
        image_width = anim_config['img_width']
        image_height = anim_config['img_height']

        aspect_ratio = image_height / image_width

        scene_width = 16  # Manim's default frame width
        scene_height = scene_width * aspect_ratio

        scale_x = scene_width / image_width*0.8
        scale_y = scene_height / image_height *0.8

        # Define translation to center the image
        # Manim's coordinate system centers at (0,0)
        # Assuming image origin is at top-left, we need to shift it
        # to center it in Manim's scene
        translation_x = -image_width / 2 * scale_x
        translation_y = image_height / 2 * scale_y

        # Function to map image coordinates to Manim coordinates
        def map_coordinates(point):
            x_img, y_img = point
            x_manim = x_img * scale_x + translation_x
            y_manim = -y_img * scale_y + translation_y  # Invert Y-axis
            return x_manim, y_manim, 0

        # Create Line objects for each fractal line
        fractal_group = VGroup()
        for line_dict in fractal_lines:
            start_pt = line_dict["start"]
            end_pt   = line_dict["end"]
            color_rgb = line_dict["color"]  # e.g. [0, 255, 255]

            start_manim = map_coordinates(start_pt)
            end_manim   = map_coordinates(end_pt)

            line_color = rgb_to_color(color_rgb)
            manim_line = Line(
                start=start_manim,
                end=end_manim,
                stroke_width=1,
                stroke_color=line_color
            )
            fractal_group.add(manim_line)

        # Optionally, add the text skeleton as a separate object for clarity
        # Uncomment if you have a separate JSON or data for the skeleton
        # Here, assuming the skeleton is already part of fractal_lines

        # Animate drawing each line one by one
        '''subgroups = split_vgroup(fractal_group, anim_config['animation_subgroup_size'])
        for subgroup in subgroups:
            self.play(Create(subgroup), run_time=1.5)'''

        # render all fractals at once instead
        self.play(Create(fractal_group), run_time=3)

        # Optionally, add a pause at the end
        self.wait(2)
