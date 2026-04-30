from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import rowcol
from PIL import Image

# ----------------------------
# Load CityJSON
# ----------------------------
def load_cityjson(path):
    with open(path) as f:
        cj = json.load(f)

    vertices = np.array(cj["vertices"])

    if "transform" in cj:
        s = np.array(cj["transform"]["scale"])
        t = np.array(cj["transform"]["translate"])
        vertices = vertices * s + t

    return cj, vertices



# ----------------------------
# Roof detection
# ----------------------------
def is_roof(coords):
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    n = np.cross(v1, v2)
    n = n / (np.linalg.norm(n) + 1e-12)
    return abs(n[2]) > 0.7


# ----------------------------
# UV from orthophoto
# ----------------------------
def roof_uv(coords, transform, w, h):
    xs = coords[:, 0]
    ys = coords[:, 1]

    rows, cols = rowcol(transform, xs, ys)

    u = np.array(cols) / w
    v = 1 - (np.array(rows) / h)

    return np.column_stack([u, v])

def build_textured_cityjson(cityjson_path, tif_path, texture_path, out_path):

    cj, vertices = load_cityjson(cityjson_path)

    src = rasterio.open(tif_path)
    transform = src.transform
    w, h = src.width, src.height

    # ----------------------------
    # GLOBAL APPEARANCE
    # ----------------------------
    appearance = {
        "textures": [
            {
                "type": "PNG",
                "image": texture_path
            }
        ],
        "materials": [
            {
                "name": "roof_material",
                "ambientIntensity": 1.0
            },
            {
                "name": "wall_gray",
                "diffuseColor": [0.6, 0.6, 0.6]
            }
        ],
        "vertices-texture": []
    }

    vertex_texture = []
    vertex_texture_map = {}

    for obj_id, obj in cj["CityObjects"].items():
        if obj["type"] != "Building":
            continue

        for geom in obj["geometry"]:

            geom["appearance"] = {
                "texture": {
                    "default": {
                        "values": []
                    }
                },
                "material": {
                    "default": {
                        "values": []
                    }
                }
            }

            # flatten faces
            if geom["type"] == "Solid":
                all_faces = [f for shell in geom["boundaries"] for f in shell]
            else:
                all_faces = geom["boundaries"]
                
            surfaces = geom.get('semantics', {}).get('surfaces', [])
            
            if len(surfaces) == len(all_faces):
                surfaces_types = [s.get('type', 'WallSurface') for s in surfaces]
                
                for face, face_type in zip(all_faces, surfaces_types):
                    ring = face[0]
                    coords = np.array([vertices[i] for i in ring])

                    if coords.shape[0] < 3:
                        continue
                    
                    uv = roof_uv(coords, transform, w, h)

                    vt_indices = []

                    for u, v in uv:
                        key = (round(float(u), 6), round(float(v), 6))

                        if key not in vertex_texture_map:
                            vertex_texture_map[key] = len(vertex_texture)
                            vertex_texture.append([float(u), float(v)])

                        vt_indices.append(vertex_texture_map[key])
                    
                    if face_type.lower().__contains__('roof'):
                        
                        geom["appearance"]["texture"]["default"]["values"].append(
                            [vt_indices]
                        )

                        # material index 0 (roof material, optional)
                        geom["appearance"]["material"]["default"]["values"].append(0)
                    else:
                        geom["appearance"]["texture"]["default"]["values"].append(None)
                        
            else:            
                for face in all_faces:

                    ring = face[0]
                    coords = np.array([vertices[i] for i in ring])

                    if coords.shape[0] < 3:
                        continue
                    
                    uv = roof_uv(coords, transform, w, h)

                    vt_indices = []

                    for u, v in uv:
                        key = (round(float(u), 6), round(float(v), 6))

                        if key not in vertex_texture_map:
                            vertex_texture_map[key] = len(vertex_texture)
                            vertex_texture.append([float(u), float(v)])

                        vt_indices.append(vertex_texture_map[key])
                    # ----------------------------
                    # ROOF → TEXTURE
                    # ----------------------------
                    if is_roof(coords):

                        geom["appearance"]["texture"]["default"]["values"].append(
                            [vt_indices]
                        )

                        # material index 0 (roof material, optional)
                        geom["appearance"]["material"]["default"]["values"].append(0)

                    # ----------------------------
                    # WALL → GRAY MATERIAL ONLY
                    # ----------------------------
                    else:
                        # NO texture
                        geom["appearance"]["texture"]["default"]["values"].append(None)

                        # material index 1 = gray
                        geom["appearance"]["material"]["default"]["values"].append(1)

    appearance["vertices-texture"] = vertex_texture
    cj["appearance"] = appearance

    with open(out_path, "w") as f:
        json.dump(cj, f)

    print("Saved:", out_path)


def save_texture(tif_path, out_dir):
    with rasterio.open(tif_path) as src:
        img = src.read([1,2,3])
        img = np.transpose(img, (1,2,0))

        # ensure uint8
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.ptp() + 1e-12) * 255).astype(np.uint8)

    out_path = out_dir + "/texture.png"
    Image.fromarray(img).save(out_path)

    return Path(out_path).absolute().__str__()


# ----------------------------
# RUN
# ----------------------------

num = 5

CITYJSON_PATH = f"OK/roof{num}.json"
TIF_PATH = f"Orthos/roof{num}.tif"
OUT_DIR = Path("Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

texture_path = save_texture(TIF_PATH, 'Outputs')
build_textured_cityjson(
    CITYJSON_PATH,
    TIF_PATH,
    texture_path,
    f"Outputs/roof{num}.json"
)
