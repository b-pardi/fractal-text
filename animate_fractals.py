from manim import *
from manim.utils.rate_functions import ease_in_out_sine

import json

with open("config.json", 'r') as f:
    anim_config = json.load(f)

# Configure Manim settings
config.frame_rate = anim_config['animation_framerate']
config.pixel_width = anim_config['animation_resolution_width']
config.pixel_height = anim_config['animation_resolution_height']
config.output_file = anim_config['animation_output_fname']
config.background_color = rgb_to_color(anim_config['animation_bg_color'])

FERN_GREEN = rgb_to_color((30,255,125))


def split_vgroup(vgroup, chunk_size):
    """Splits a VGroup into a list of smaller VGroups each containing chunk_size elements."""
    return [VGroup(*vgroup[i:i + chunk_size]) for i in range(0, len(vgroup), chunk_size)]

class FractalTextAnimation(Scene):
    def construct(self):
        global anim_config
        
        # Load fractal lines from JSON file
        with open("output/fractal_lines.json", "r") as f:
            all_fractals = json.load(f)

        # Define scaling factors based on image dimensions
        image_width = anim_config['img_width']
        image_height = anim_config['img_height']

        aspect_ratio = image_height / image_width

        scene_width = 16
        scene_height = scene_width * aspect_ratio

        scale_x = scene_width / image_width*0.8
        scale_y = scene_height / image_height *0.8

        # Define translation to center the image
        # Manim's coordinate system centers at (0,0)
        # image origin is at top-left, we need to shift it to center it in Manim's scene
        translation_x = -image_width / 2 * scale_x
        translation_y = image_height / 2 * scale_y

        # Function to map image coordinates to Manim coordinates
        def map_coordinates(point):
            x_img, y_img = point
            x_manim = x_img * scale_x + translation_x
            y_manim = -y_img * scale_y + translation_y  # Invert Y-axis
            return x_manim, y_manim, 0

        max_iterations = max(len(fractal_iterations) for fractal_iterations in all_fractals)

        # iteration animations for a Succession
        iteration_animations = []
        all_fractals_group = VGroup()
        for i in range(max_iterations):
            # sub_anims for iteration i across all fractals in parallel
            sub_anims = []

            for fractal_idx, fractal_iterations in enumerate(all_fractals):
                if i >= len(fractal_iterations):
                    # fractal has fewer iterations than i
                    continue

                # fractal_iterations[i] is a list of line dicts for iteration i
                iteration_line_dicts = fractal_iterations[i]

                # Build a VGroup for these lines
                vgroup = VGroup()
                for line_dict in iteration_line_dicts:
                    start_pt = line_dict["start"]
                    end_pt   = line_dict["end"]
                    color_rgb= line_dict["color"]

                    start_m = map_coordinates(start_pt)
                    end_m   = map_coordinates(end_pt)
                    line_col= rgb_to_color(color_rgb)

                    line_obj = Line(
                        start=start_m,
                        end=end_m,
                        stroke_width=1,
                        stroke_color=line_col
                    )
                    vgroup.add(line_obj)
                all_fractals_group.add(vgroup)

                # "Create" all lines for fractal_idx's iteration i in parallel
                sub_anims.append(Create(vgroup, rate_func=linear))

            # turn sub_anims into a single parallel animation for iteration i
            iteration_anim = AnimationGroup(*sub_anims, lag_ratio=0.0, run_time=1)

            # add iteration_anim to the overall timeline
            iteration_animations.append(iteration_anim)

        # Build a Succession so iteration i starts after iteration i-1 ends.
        # so we have one big timeline from start to finish 
        entire_fractal_animation = Succession(*iteration_animations)

        # 5) Animate the entire fractal with an overall run_time & rate_func
        self.play(entire_fractal_animation, run_time=20, rate_func=smooth)

        self.wait(2)
        self.play(FadeOut(all_fractals_group), run_time=2)

class BarnsleyFernHorizontal(Scene):
    def construct(self):
        global anim_config

        # load fractals from JSON
        with open("output/barnsley_fern_points.json", "r") as f:
            all_ferns = json.load(f)
            # all_ferns[0], all_ferns[1], all_ferns[2]
            # each is a list of (x,y)

        # Combine bounding box for all ferns
        all_x = []
        all_y = []
        for i in range(3):
            for (x, y) in all_ferns[i]:
                all_x.append(x)
                all_y.append(y)

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        range_x = max_x - min_x
        range_y = max_y - min_y

        if range_x == 0:
            range_x = 1
        if range_y == 0:
            range_y = 1

        # padding
        min_x -= 0.1 * range_x
        max_x += 0.1 * range_x
        min_y -= 0.1 * range_y
        max_y += 0.15 * range_y

        # Update range_x, range_y after padding
        range_x = max_x - min_x
        range_y = max_y - min_y

        # Mapping function: let's map
        #   [min_x..max_x] -> [-8..8], [min_y..max_y] -> [-4.5..4.5]
        #   for a 16:9 shape in Manim's default coords
        def points_to_np_array(pts):
            arr = []
            for (x, y) in pts:
                x_norm = (x - min_x) / range_x  # in [0..1], plus padding
                y_norm = (y - min_y) / range_y
                X = x_norm * 16 - 8.0
                Y = y_norm * 9  - 4.5
                arr.append([X, Y, 0])
            return np.array(arr)

        # Convert each fractal's points
        arr0 = points_to_np_array(all_ferns[0])
        arr1 = points_to_np_array(all_ferns[1])
        arr2 = points_to_np_array(all_ferns[2])

        assert(len(arr0) == len(arr1) == len(arr2))

        def make_vgroup_of_dots(np_arr):
            return VGroup(
                *[
                    Dot(point=p, radius=0.01, color=FERN_GREEN)
                    for p in np_arr
                ]
            )

        # Build the three fractals as VGroups
        cloud0 = make_vgroup_of_dots(arr0)
        cloud1 = make_vgroup_of_dots(arr1)
        cloud2 = make_vgroup_of_dots(arr2)

        # Chunked fade-in for the first fractal (cloud0)
        fade_time = 8
        total_points = len(cloud0)
        num_subgroups = 500 # e.g. 40 steps in the fade
        chunk_size = int(np.ceil(total_points / num_subgroups))

        subgroups = [
            VGroup(*cloud0[i : i+chunk_size])
            for i in range(0, total_points, chunk_size)
        ]

        run_time_sub = fade_time / num_subgroups
        fadein_anims = [FadeIn(sg, run_time=run_time_sub) for sg in subgroups]
        fade_in_sequence = Succession(*fadein_anims)

        # Play the fade-in sequence so fractal #0's points appear gradually
        self.play(fade_in_sequence)

        self.play(Transform(cloud0, cloud2), run_time=3)
        self.play(Transform(cloud0, cloud1), run_time=3)
        self.play(Transform(cloud0, cloud2), run_time=3)
        self.play(Transform(cloud0, cloud1), run_time=3)
        self.play(Transform(cloud0, cloud2), run_time=3)
        self.play(Transform(cloud0, make_vgroup_of_dots(arr0)), run_time=3)

        # 9) Pause
        self.wait(1)
        self.play(FadeOut(cloud0), run_time=2)