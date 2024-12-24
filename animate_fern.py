from manim import *
import json

with open("config.json", 'r') as f:
    anim_config = json.load(f)

config.frame_rate = anim_config['animation_framerate']
config.pixel_width = anim_config['animation_resolution_width']
config.pixel_height = anim_config['animation_resolution_height']
config.output_file = 'fern'
#config.background_color = config  # Optional: Set background color


class BarnsleyFernHorizontal(Scene):
    def construct(self):
        global anim_config
        # 1) Load horizontally oriented points
        with open("output/barnsley_fern_points.json", "r") as f:
            fern_points = json.load(f)

        # 2) Build bounding box
        xs = [p[0] for p in fern_points]
        ys = [p[1] for p in fern_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        range_x = max_x - min_x if max_x != min_x else 1
        range_y = max_y - min_y if max_y != min_y else 1

        # 3) Create a group of Dots (or a DotCloud)
        fern_group = VGroup()
        
        # Subsample for performance
        skip = 500
        for (px, py) in fern_points[::skip]:
            # Normalize to [0..1]
            x_norm = (px - min_x) / range_x
            y_norm = (py - min_y) / range_y
            # Let’s map them to [-6..+6] in X, [-3..+3] in Y, for a wide shape
            X = x_norm*12 - 6
            Y = y_norm*6 - 3  # half the range in Y to keep it wide
            dot = Dot(point=[X, Y, 0], radius=0.01, color=GREEN)
            fern_group.add(dot)

        # 4) Animate drawing the fern
        #    Option A: Show all at once
        # self.play(FadeIn(fern_group), run_time=2)

        #    Option B: Show in small subgroups
        chunk_size = 50
        subgroups = [VGroup(*fern_group[i:i+chunk_size])
                     for i in range(0, len(fern_group), chunk_size)]

        for subgroup in subgroups:
            self.play(FadeIn(subgroup), run_time=0.1)

        # 5) Apply a left-right shear to “sway” the fern
        def shear_matrix(k):
            return [
                [1, k, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]

        # E.g. do 0.3 to -0.3
        self.play(fern_group.animate.apply_matrix(shear_matrix(0.3)), run_time=1)
        self.play(fern_group.animate.apply_matrix(shear_matrix(-0.3)), run_time=1)
        self.play(fern_group.animate.apply_matrix(shear_matrix(0.0)), run_time=1)

        # 6) Pause
        self.wait(2)
