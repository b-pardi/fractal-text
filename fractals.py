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
        font = ImageFont.load_default('arial.ttf', font_size)

    # Split text into lines
    lines = text.split('\n')

    # Measure total height of text block
    line_height = font.getsize("A")[1]
    total_height = len(lines) * line_height

    # Starting y position to center vertically
    y_offset = (height - total_height) // 2

    for line in lines:
        # Measure width of each line
        line_width = draw.textsize(line, font=font)[0]
        # Center each line horizontally
        x_offset = (width - line_width) // 2
        # Draw the line
        draw.text((x_offset, y_offset*0.8), line, font=font, fill=255)
        y_offset += line_height

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
def dragon_curve(iterations=10, step=5):
    """
    Generate a Dragon Curve path.
    Axiom: FX
    X->X+YF+
    Y->-FX-Y
    F-> forward
    +-> turn left 90
    --> turn right 90
    """
    rules = {
        'X': "X+YF+",
        'Y': "-FX-Y"
    }
    s = "FX"
    for _ in range(iterations):
        new_s = []
        for c in s:
            new_s.append(rules.get(c, c))
        s = "".join(new_s)
    # Convert to line segments
    return string_to_lines(s, step=step, angle_increment=90)

def hilbert_curve(order=4, step=5):
    """
    Hilbert Curve:
    Axiom: A
    A->+BF−AFA−FB+
    B->−AF+BFB+FA−
    """
    rules = {
        'A': "+BF-AFA-FB+",
        'B': "-AF+BFB+FA-"
    }
    s = "A"
    for _ in range(order):
        new_s = "".join(rules.get(c, c) for c in s)
        s = new_s
    return string_to_lines(s, step=step, angle_increment=90)

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

if __name__ == "__main__":
    text = "Brandon\n\u2661\nJordyn"
    frame_dir = "temp_frames/"
    print(text)
    # Small font & skeletonize
    skel = text_to_skeleton_array(text, width=6200, height=5000, font_path='arial unicode ms.otf', font_size=1600)
    skeleton_pixels = find_skeleton_pixels(skel)

    # Pick which fractal to use:
    lines_fractal = dragon_curve(iterations=8, step=12)  # Dragon Curve
    #lines_fractal = hilbert_curve(order=5, step=3)  # Hilbert Curve

    final_arr = np.copy(skel)
    all_fractal_lines  = []

    num_points = 5000
    chosen_points = grid_based_sampling(skeleton_pixels, num_points)

    for idx, (y, x) in enumerate(chosen_points):
        y, x = int(y), int(x)
        angle = compute_outward_direction(skel, y, x)
        # Rotate fractal lines by 'angle'
        rotated_lines = rotate_lines(lines_fractal, angle)
        # Place fractal at (x, y)
        place_lines_in_image(final_arr, rotated_lines, start_x=x, start_y=y)
        # Save the current state as a frame
        for (x1, y1, x2, y2) in rotated_lines:
            # Translate to absolute positions
            abs_x1 = x1 + x
            abs_y1 = y1 + y
            abs_x2 = x2 + x
            abs_y2 = y2 + y
            all_fractal_lines.append([[abs_x1, abs_y1], [abs_x2, abs_y2]])
        if (idx+1) % 100 == 0 or (idx+1) == num_points:
            print(f"Placed {idx+1}/{num_points} fractals.")

    final_img = Image.fromarray((final_arr*255).astype(np.uint8), mode='L')
    final_img.save("output/thin_text_with_fractals.png", dpi=(400,400))
    print("Saved thin_text_with_fractals.png")

    fractal_output_file = "output/fractal_lines.json"
    save_fractal_lines(all_fractal_lines, fractal_output_file)
