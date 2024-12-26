import argparse
import json

from src.fractal_text import fractal_text
from src.fern import barnsley_fern
from src.enums import FractalType

def main():
    parser = argparse.ArgumentParser(description="Generate fractals.")

    # choose which fractal to generate
    parser.add_argument(
        "fractal",
        choices=["dragon", "hilbert", "barnsley"],
        help="Type of fractal to generate (dragon, hilbert, barnsley)."
    )
    
    # text argument for dragon or hilbert
    parser.add_argument(
        "--text",
        type=str,
        help="Text to generate fractal off of (required for dragon and hilbert)."
    )
    
    args = parser.parse_args()
    
    # Validate that `text` is required for dragon or hilbert
    if args.fractal in ["dragon", "hilbert"] and not args.text:
        parser.error(f"--text is required when fractal is '{args.fractal}'.")

    with open("config.json", 'r') as f:
        config = json.load(f)

    print(args.fractal)
    if args.fractal == FractalType.BARNSLEY.value:
        barnsley_fern(config)
        print("ASDF")
    else:
        fractal_text(args.text, FractalType(args.fractal), config)


if __name__ == '__main__':
    main()