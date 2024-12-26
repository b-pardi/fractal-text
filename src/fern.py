import numpy as np
import json
from PIL import Image


def generate_barnsley_fern(num_points=20000, scale=100, f2_x_scale=0.85):
    """
    Generates Barnsley fern points.
    Returns a list of (x, y) in floating coords.
    
    'scale' is how large the fractal’s coordinates become if you want
    them in pixel space. Tweak as needed.
    """
    np.random.seed(42)
    # Barnsley Fern Transform parameters
    # Probability intervals + transform definitions
    # f1
    p1 = 0.01
    f1 = lambda x, y: (0, 0.16*y)
    # f2
    p2 = 0.85
    f2 = lambda x, y: (f2_x_scale*x + 0.04*y, -0.04*x + 0.85*y + 1.6)
    # f3
    p3 = 0.07
    f3 = lambda x, y: (0.20*x - 0.26*y, 0.23*x + 0.22*y + 1.6)
    # f4
    p4 = 0.07
    f4 = lambda x, y: (-0.15*x + 0.28*y, 0.26*x + 0.24*y + 0.44)

    points = []
    x, y = 0.0, 0.0  # Starting point

    for _ in range(num_points):
        r = np.random.random()
        if r < p1:
            x, y = f1(x, y)
        elif r < p1 + p2:
            x, y = f2(x, y)
        elif r < p1 + p2 + p3:
            x, y = f3(x, y)
        else:
            x, y = f4(x, y)
        points.append((x*scale, y*scale))
    return points

def render_fern_image(points, width=1200, height=1200, bg_color=(0,0,0), fg_color=(0,255,0)):
    """
    Renders the Barnsley fern points to a PIL Image of given width/height.
    bg_color / fg_color are (R,G,B) in 0..255
    """
    img = Image.new("RGB", (width, height), bg_color)
    # Convert to a 2D pixel array
    pix = img.load()

    # Find bounding box of the points to scale them to the image
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Pad a little so the fern isn’t touching the edges
    pad = 20
    range_x = range_x if range_x != 0 else 1
    range_y = range_y if range_y != 0 else 1

    for (px, py) in points:
        # map to [0..width], [0..height]
        x_norm = (px - min_x) / range_x
        y_norm = (py - min_y) / range_y
        # invert y if needed so it draws upright
        x_img = int(x_norm * (width - 2*pad) + pad)
        y_img = int((1 - y_norm) * (height - 2*pad) + pad)
        if 0 <= x_img < width and 0 <= y_img < height:
            pix[x_img, y_img] = fg_color

    return img

def barnsley_fern(config):    
    all_fern_ponts = []

    # shearing x of f2 in 3 different stages to animate a flow between them
    # shows the fern sort of waving around
    x_scales = [0.85, 0.68, 0.95]
    for f2_x in x_scales:
        # generate points
        fern_points = generate_barnsley_fern(num_points=200000, scale=100, f2_x_scale=f2_x)

        # rotate horizontally for widescreen display
        fern_points = [(y,-x) for (x, y) in fern_points]
        
        out_img = render_fern_image(
            fern_points,
            width=1280,
            height=720,
            bg_color=tuple(config['output_img_bg_color']),
            fg_color=(30,255,125)
        )
        out_img.save(f"output/{config['output_generation_name']}-{f2_x}.png")
        print("Saved barnsley fern")
        all_fern_ponts.append(fern_points)

    # save points to JSON for Manim, stored a list of [x, y]
    with open(f"output/{config['output_generation_name']}.json", "w") as f:
        json.dump(all_fern_ponts, f)
    print("Saved barnsley fern points for animation")

