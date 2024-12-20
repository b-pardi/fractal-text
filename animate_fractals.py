from manim import *
import json
import math

class FractalTextAnimation(Scene):
    def construct(self):
        # Load fractal lines from JSON file
        with open("output/fractal_lines.json", "r") as f:
            fractal_lines = json.load(f)

        # Define scaling factors based on image dimensions
        image_width = 6000
        image_height = 6000
        scene_width = 14  # Manim's default frame width
        scene_height = 14  # Keeping it square for simplicity

        scale_x = scene_width / image_width
        scale_y = scene_height / image_height

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
            y_manim = y_img * scale_y + translation_y
            return x_manim, y_manim, 0  # Manim uses 3D coordinates

        # Group to hold all line segments
        fractal_group = VGroup()

        # Create Line objects for each fractal line
        for line in fractal_lines:
            start, end = line
            start_manim = map_coordinates(start)
            end_manim = map_coordinates(end)
            manim_line = Line(start=start_manim, end=end_manim, stroke_width=1, stroke_color=WHITE)
            fractal_group.add(manim_line)

        # Optionally, add the text skeleton as a separate object for clarity
        # Uncomment if you have a separate JSON or data for the skeleton
        # Here, assuming the skeleton is already part of fractal_lines

        # Animate drawing each line one by one
        for line in fractal_group:
            self.play(Create(line), run_time=0.01)  # Adjust run_time for speed

        # Optionally, add a pause at the end
        self.wait(2)
