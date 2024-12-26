# Fractal Text Generator

A Python-based fractal generator that produces visualizations of text-based and mathematical fractals, including **Dragon Curve**, **Hilbert Curve**, and **Barnsley Fern**. The project also supports animation rendering and skeletonized text-based fractals with customizable options.

## Features
- Generate fractals of three types: **Dragon Curve**, **Hilbert Curve**, and **Barnsley Fern**.
- Create visually appealing fractals using skeletonized text as a base for Dragon and Hilbert curves.
- customizable through a configuration file (`config.json`).
- Supports animation rendering of fractals with adjustable parameters.
- Outputs fractals as high-resolution images or animations.

---

## Installation

1. Install dependencies (recommended to make a venv first)
`pip install -r requirements.txt`

2. Font(s)
- Download your font of choice for the skeleton text
    - [Google Fonts](https://fonts.google.com/) or [DaFont](https://www.dafont.com/)
    - **Note** If you want special chars like ❤ (\u2764) then chose a font with unicode support such as the one already in the fonts folder
- place `.otf` or `.ttf` in the fonts folder
- update the config.json with the filename of your chosen font

**Note** below steps only required for anmation, if just generating images these can be skipped

3. Install [FFmpeg](https://ffmpeg.org/)

4. Install latex:
- windows: [miktex](https://miktex.org/https://ffmpeg.org/)
- linux: `sudo apt update && sudo apt install texlive-full`

5. verify: 
- `ffmpeg -version`
- `latex --version`



## Usage

**First generate a fractal with `main.py` and then animate with manim script**

### Generate a Fractal
To generate a fractal, use the following command:

`python main.py <fractal_type> --text "Your Text Here"`

### Configuration
See the `config.json` to tweak parameters of generation to your liking

### Arguments
<fractal>: Specify the type of fractal to generate. Options:
- dragon
- hilbert
- barnsley (not for text generation)

--text: (Required for dragon and hilbert) Text to use as a skeleton for fractal generation.

### Examples
Generate a Dragon Curve fractal based on text:
`python main.py dragon --text "Fractal Magic"`

Generate a Hilbert Curve fractal based on text:
`python main.py hilbert --text "Hilbert Rocks"`

Generate a Barnsley Fern fractal:
`python main.py barnsley`

### Animating the generation
To use manim we do not just call the script directly, use the commands below:

- To generate animation for text fractals
`manim animate_fractals.py FractalTextAnimation`

- To generate animation for Barnsley Fern
`manim animate_fractals.py BarnsleyFernHorizontal`

optional, `--disable_caching` if little to no repetition in animations

## Output
### Image Output
Generated fractals are saved as .png images in the output/ directory. Each fractal is named using the output_generation_name parameter in config.json.

### Animation Output
For animated fractals, the output is saved as an .mp4 file with the specified resolution and framerate in the media folder.