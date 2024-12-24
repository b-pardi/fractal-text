import math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
import imageio
import json
import os

def text_to_skeleton_array(text, width=300, height=100, font_path=None, font_size=20):
    """
    Render text and skeletonize it to get a thin line representation.
    """
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Split text into lines
    lines = text.split('\n')

    # Measure total height of text block
    sample_bbox = font.getbbox("A")
    line_height = sample_bbox[3] - sample_bbox[1]
    total_height = len(lines) * line_height

    # Starting y position to center vertically
    y_offset = (height - total_height*1.54) // 2

    for line in lines:
        # Measure width of each line
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        # Center each line horizontally
        x_offset = (width - line_width) // 2
        # Draw the line
        draw.text((x_offset, y_offset*0.8), line, font=font, fill=255)
        y_offset += line_height*1.54

    # Convert to binary array
    arr = np.array(img)
    binary = (arr > 128).astype(np.uint8)

    # Skeletonize the text to reduce it to a single-pixel-wide line
    skeleton = skeletonize(binary)
    skeleton_arr = skeleton.astype(np.uint8)

    return skeleton_arr

def find_skeleton_pixels(skel_arr):
    """
    Find all pixels in the skeleton line.
    """
    coords = np.argwhere(skel_arr == 1)
    return coords

def compute_outward_direction(binary, y, x):
    """
    Compute an outward direction from a line pixel.
    We'll look for background pixels around and point that way.
    With a very thin line, just pick a direction perpendicular to the line direction.
    We'll estimate direction by looking at neighbors.
    """
    h, w = binary.shape
    radius = 2
    # Identify the line direction by finding nearby skeleton pixels and computing a direction vector
    neighbors = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w:
                if binary[ny,nx] == 1:
                    neighbors.append((dx, dy))
    # If we have neighbors, find the average direction of them to determine line direction
    if len(neighbors) > 0:
        avg_dx = np.mean([p[0] for p in neighbors])
        avg_dy = np.mean([p[1] for p in neighbors])
        # Line direction angle
        line_angle = math.degrees(math.atan2(-avg_dy, avg_dx))
        # Outward direction: pick perpendicular direction
        # We can choose perpendicular by adding 90 degrees or -90 degrees
        # Check which side is more background:
        # Sample a point perpendicular to line direction +90 deg
        perp_angle_1 = line_angle + 90
        perp_angle_2 = line_angle - 90

        def check_bg(angle):
            rad = math.radians(angle)
            test_x = x + 5*math.cos(rad)
            test_y = y - 5*math.sin(rad)
            # count how many bg pixels near test_x,test_y
            count_bg = 0
            for ty in range(-2,3):
                for tx in range(-2,3):
                    yy = int(test_y+ty)
                    xx = int(test_x+tx)
                    if 0 <= yy < h and 0 <= xx < w:
                        if binary[yy,xx] == 0:
                            count_bg += 1
            return count_bg

        bg_count_1 = check_bg(perp_angle_1)
        bg_count_2 = check_bg(perp_angle_2)

        # Choose direction with more background
        if bg_count_1 > bg_count_2:
            return perp_angle_1
        else:
            return perp_angle_2
    else:
        # No neighbors? just pick a random angle
        return random.uniform(0,360)


# -------- Fractal Generators --------
def dragon_curve_partial(iterations=10, step=5):
    """
    Return a list of line sets, where partial_lines_per_iteration[i]
    contains only the *newly added* lines introduced during iteration i.

    Axiom: "FX"
    X -> X+YF+
    Y -> -FX-Y
    """
    rules = {
        'X': "X+YF+",
        'Y': "-FX-Y"
    }
    s = "FX"

    partial_lines_per_iteration = []
    old_lines = []  # lines from the previous iteration (start empty)

    for i in range(iterations):
        # Expand string once
        new_s = []
        for c in s:
            new_s.append(rules.get(c, c))
        s = "".join(new_s)

        # Convert the entire string so far into line segments
        all_lines = string_to_lines(s, step=step, angle_increment=90)

        # Build sets of float-rounded line segments for easy 'difference' comparison
        new_set = set(
            (round(x1,2), round(y1,2), round(x2,2), round(y2,2))
            for (x1, y1, x2, y2) in all_lines
        )
        old_set = set(
            (round(x1,2), round(y1,2), round(x2,2), round(y2,2))
            for (x1, y1, x2, y2) in old_lines
        )

        # Difference = lines that appear in iteration i but not before
        diff_set = new_set - old_set

        # Build a list of actual floating line segments corresponding to the difference
        diff_lines = []
        for seg in all_lines:
            seg_rounded = (
                round(seg[0],2), round(seg[1],2),
                round(seg[2],2), round(seg[3],2)
            )
            if seg_rounded in diff_set:
                diff_lines.append(seg)

        partial_lines_per_iteration.append(diff_lines)

        # Update old_lines for the next iteration
        old_lines = all_lines

    return partial_lines_per_iteration


def hilbert_curve_partial(order=4, step=5):
    """
    Return a list of line sets, where partial_lines_per_iteration[i]
    contains only the *newly added* lines introduced during iteration i.

    Axiom: "A"
    Rules:
      A -> +BF-AFA-FB+
      B -> -AF+BFB+FA-
    """
    rules = {
        'A': "+BF-AFA-FB+",
        'B': "-AF+BFB+FA-"
    }
    s = "A"

    partial_lines_per_iteration = []
    old_lines = []

    for i in range(order):
        # Expand string once
        new_s = []
        for c in s:
            new_s.append(rules.get(c, c))
        s = "".join(new_s)

        # Convert the entire string so far into line segments
        all_lines = string_to_lines(s, step=step, angle_increment=90)

        # Compare sets
        new_set = set(
            (round(x1,2), round(y1,2), round(x2,2), round(y2,2))
            for (x1, y1, x2, y2) in all_lines
        )
        old_set = set(
            (round(x1,2), round(y1,2), round(x2,2), round(y2,2))
            for (x1, y1, x2, y2) in old_lines
        )

        diff_set = new_set - old_set
        diff_lines = []
        for seg in all_lines:
            seg_rounded = (
                round(seg[0],2), round(seg[1],2),
                round(seg[2],2), round(seg[3],2)
            )
            if seg_rounded in diff_set:
                diff_lines.append(seg)

        partial_lines_per_iteration.append(diff_lines)

        old_lines = all_lines

    return partial_lines_per_iteration


def grid_based_sampling(skeleton_pixels, num_points):
    """
    Evenly distribute points by dividing the image into a grid and selecting one point per grid cell.
    """
    if len(skeleton_pixels) == 0:
        return np.array([])

    h, w = skeleton_pixels.max(axis=0)[0] + 1, skeleton_pixels.max(axis=0)[1] + 1

    # Define grid size based on number of points
    grid_size = int(math.sqrt(num_points))
    if grid_size == 0:
        grid_size = 1
    step_y = h // grid_size
    step_x = w // grid_size

    chosen_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Define the grid cell boundaries
            y_start = i * step_y
            y_end = (i + 1) * step_y if i < grid_size -1 else h
            x_start = j * step_x
            x_end = (j + 1) * step_x if j < grid_size -1 else w

            # Find points within the grid cell
            points_in_cell = skeleton_pixels[
                (skeleton_pixels[:,0] >= y_start) & (skeleton_pixels[:,0] < y_end) &
                (skeleton_pixels[:,1] >= x_start) & (skeleton_pixels[:,1] < x_end)
            ]

            if len(points_in_cell) > 0:
                # Randomly select one point from the cell
                idx = random.randint(0, len(points_in_cell) -1)
                chosen_points.append(points_in_cell[idx])

            if len(chosen_points) >= num_points:
                break
        if len(chosen_points) >= num_points:
            break

    return np.array(chosen_points[:num_points])

def string_to_lines(s, step=5, angle_increment=90, initial_angle=0):
    direction = initial_angle
    x, y = 0.0, 0.0
    lines = []
    stack = []
    for c in s:
        if c == 'F':
            rad = math.radians(direction)
            x2 = x + step*math.cos(rad)
            y2 = y - step*math.sin(rad)
            lines.append((x, y, x2, y2))
            x, y = x2, y2
        elif c == '+':
            direction += angle_increment
        elif c == '-':
            direction -= angle_increment
        elif c == '[':
            stack.append((x,y,direction))
        elif c == ']':
            x,y,direction = stack.pop()
        # X,Y etc are no-op except as rules expansions
    return lines

def rotate_line(x, y, angle):
    """
    Rotate a point (x, y) around the origin by 'angle' degrees.
    """
    rad = math.radians(angle)
    cosA = math.cos(rad)
    sinA = math.sin(rad)
    rx = x * cosA - y * sinA
    ry = x * sinA + y * cosA
    return rx, ry

def rotate_lines(lines, angle):
    """
    Rotate all lines by 'angle' degrees.
    """
    rotated = []
    for (x1, y1, x2, y2) in lines:
        rX1, rY1 = rotate_line(x1, y1, angle)
        rX2, rY2 = rotate_line(x2, y2, angle)
        rotated.append((rX1, rY1, rX2, rY2))
    return rotated

# Drawing lines on array
def draw_line_on_array(arr, x1, y1, x2, y2):
    height, width = arr.shape
    # Bresenham
    dx = abs(x2 - x1)
    sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1)
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    x, y = x1, y1
    while True:
        if 0 <= x < width and 0 <= y < height:
            arr[y, x] = 1
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

def place_lines_in_image(base_array, lines, start_x, start_y):
    for (x1, y1, x2, y2) in lines:
        draw_line_on_array(base_array, 
                           int(x1 + start_x), int(y1 + start_y), 
                           int(x2 + start_x), int(y2 + start_y))

def draw_line_on_array_color(arr, x1, y1, x2, y2, color):
    """
    Bresenham in RGB array. color is (R,G,B).
    """
    height, width, _ = arr.shape
    dx = abs(x2 - x1)
    sx = 1 if x1 < x2 else -1
    dy = -abs(y2 - y1)
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    x, y = x1, y1

    while True:
        if 0 <= x < width and 0 <= y < height:
            arr[y, x] = color
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

def place_lines_in_image_color(base_array, lines, start_x, start_y, color):
    """
    Place lines into an RGB array, applying the specified color.
    """
    for (lx1, ly1, lx2, ly2) in lines:
        abs_x1 = int(lx1 + start_x)
        abs_y1 = int(ly1 + start_y)
        abs_x2 = int(lx2 + start_x)
        abs_y2 = int(ly2 + start_y)
        draw_line_on_array_color(base_array, abs_x1, abs_y1, abs_x2, abs_y2, color)

def save_fractal_lines(fractal_lines, output_filename):
    """
    fractal_lines is a list of dicts, each with
      "start": [x1, y1],
      "end":   [x2, y2],
      "color": [r, g, b]
    """
    import json
    with open(output_filename, 'w') as f:
        json.dump(fractal_lines, f, indent=2)
    print(f"Fractal lines saved to {output_filename}")

def save_fractal_lines(fractal_lines, output_filename):
    """
    Save fractal lines to a JSON file.
    Each line is represented as a tuple of start and end coordinates.
    """
    with open(output_filename, 'w') as f:
        json.dump(fractal_lines, f)
    print(f"Fractal lines saved to {output_filename}")

def load_fractal_lines(input_filename):
    """
    Load fractal lines from a JSON file.
    Returns a list of line segments.
    """
    with open(input_filename, 'r') as f:
        lines = json.load(f)
    return lines

def interpolate_color(c1, c2, t):
    """
    Interpolate between color c1 and c2 with factor t in [0,1].
    c1, c2 are (R,G,B) tuples, t is float in [0,1].
    """
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return (r, g, b)

def custom_color_gradient(fraction):
    """
    Given a fraction in [0,1], return an (R,G,B) by interpolating
    across multiple color stops. For example:
    0.0 --> blue
    0.4 --> green
    1.0 --> red
    
    The stops are a list of (stop_fraction, (R,G,B)).
    We linearly interpolate between whichever two stops we are between.
    """
    color_stops = [
        (0.4, (0, 255, 255)),  # fraction=0  --> start color
        #(0.75, (250, 250, 250)),  # fraction=0.5  --> mid color
        (1.0, (50, 50, 255)),  # fraction=1.0 --> end color
    ]

    # If fraction is less than the 1st stop, clamp to first color
    if fraction <= color_stops[0][0]:
        return color_stops[0][1]
    # If fraction is beyond the last stop, clamp to last color
    if fraction >= color_stops[-1][0]:
        return color_stops[-1][1]

    # Otherwise find which two stops this fraction is between
    for i in range(len(color_stops) - 1):
        f0, c0 = color_stops[i]
        f1, c1 = color_stops[i+1]
        if f0 <= fraction <= f1:
            # Map fraction from [f0,f1] to [0,1]
            t = (fraction - f0) / (f1 - f0)
            return interpolate_color(c0, c1, t)

    # Fallback
    return (255, 255, 255)

def get_gradient_color_from_iteration(iter_index, total_iterations):
    if total_iterations <= 1:
        return (255, 255, 255)

    fraction = iter_index / (total_iterations - 1)
    return custom_color_gradient(fraction)

if __name__ == "__main__":
    with open("config.json", 'r') as f:
        config = json.load(f)

    text = config['text']
    
    # 1) Skeletonize text (unchanged)
    skel = text_to_skeleton_array(
        text, 
        width=config['img_width'], 
        height=config['img_height'], 
        font_path=config['font_path'], 
        font_size=config['font_size']
    )
    skeleton_pixels = find_skeleton_pixels(skel)

    # 2) Pick fractal approach, but use *partial* version
    if config['fractal_type'] == 'dragon':
        fractal_iterations = dragon_curve_partial(
            iterations=config['iters'], 
            step=config['steps']
        )
    elif config['fractal_type'] == 'hilbert':
        fractal_iterations = hilbert_curve_partial(
            order=config['order'], 
            step=config['steps']
        )
    else:
        raise ValueError("Unsupported fractal type")

    # Prepare an RGB array
    final_arr = np.zeros((config['img_height'], config['img_width'], 3), dtype=np.uint8)
    all_fractal_lines = []

    # 3) Chosen skeleton anchor points
    num_points = config['num_src_pts']
    chosen_points = grid_based_sampling(skeleton_pixels, num_points)

    # 4) For each chosen skeleton point, place each iteration's *new lines* in its iteration color
    for idx, (y, x) in enumerate(chosen_points):
        angle = compute_outward_direction(skel, y, x)

        for iter_index, lines_this_iter in enumerate(fractal_iterations):
            color = get_gradient_color_from_iteration(
                iter_index,                # iteration index
                config['iters']    # total iterations
            )

            # Rotate only the newly-added lines of this iteration
            rotated_lines = rotate_lines(lines_this_iter, angle)

            # Place fractal lines at (x, y)
            place_lines_in_image_color(
                final_arr, 
                rotated_lines, 
                start_x=x, 
                start_y=y, 
                color=color
            )

            # Store lines in JSON with color
            for (lx1, ly1, lx2, ly2) in rotated_lines:
                all_fractal_lines.append({
                    "start": [lx1 + x, ly1 + y],
                    "end":   [lx2 + x, ly2 + y],
                    "color": list(color)
                })

        if (idx+1) % 10 == 0 or (idx+1) == num_points:
            print(f"Placed fractals at {idx+1}/{num_points} skeleton points.")

    # 5) Save the final image
    final_img = Image.fromarray(final_arr, mode='RGB')
    final_img.save("output/thin_text_with_fractals_color.png", dpi=(400,400))
    print("Saved thin_text_with_fractals_color.png")

    # 6) Save lines (with iteration color) to JSON
    fractal_output_file = "output/fractal_lines.json"
    save_fractal_lines(all_fractal_lines, fractal_output_file)
