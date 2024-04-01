import argparse
import os

# Arguments
parser = argparse.ArgumentParser(description='Convert Metafile to COCO format')
parser.add_argument('--metafile', type=str, required=True, help='Path to the metafile')
args = parser.parse_args()