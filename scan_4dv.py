
import os

def scan_4dv(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.vol'):
                path = os.path.join(root, file)
                size = os.path.getsize(path)
                if size % (256*256) == 0:
                    z = size // (256*256)
                    print(f"Found match: {path}, Size: {size}, Z: {z}")
                elif size % (128*128) == 0:
                    z = size // (128*128)
                    print(f"Found match (128x128): {path}, Size: {size}, Z: {z}")

if __name__ == "__main__":
    scan_4dv('/data1/sunchao/CQH/data/胎儿脊柱/4dv')
