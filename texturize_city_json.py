from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import rowcol
from PIL import Image


# ============================================================
# LOAD CITYJSON + APPLY TRANSFORM
# ============================================================
def load_cityjson(path):
    with open(path) as f:
        cj = json.load(f)

    vertices = np.array(cj["vertices"])

    if "transform" in cj:
        s = np.array(cj["transform"]["scale"])
        t = np.array(cj["transform"]["translate"])
        vertices = vertices * s + t

    return cj, vertices


# ============================================================
# GEOMETRY HELPERS
# ============================================================
def extract_faces(geom):
    if geom["type"] == "Solid":
        return [f for shell in geom["boundaries"] for f in shell]
    return geom["boundaries"]


def get_face_coords(face, vertices):
    outer_ring = face[0]
    coords = np.array([vertices[i] for i in outer_ring])
    return coords


# ============================================================
# ROOF DETECTION
# ============================================================
def is_roof(coords):
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    n = np.cross(v1, v2)
    n = n / (np.linalg.norm(n) + 1e-12)
    return abs(n[2]) > 0.7


def is_roof_semantic(face_type):
    return "roof" in face_type.lower()


# ============================================================
# UV COMPUTATION
# ============================================================
def compute_uv(coords, transform, w, h):
    xs = coords[:, 0]
    ys = coords[:, 1]

    rows, cols = rowcol(transform, xs, ys)

    u = np.array(cols) / w
    v = 1 - (np.array(rows) / h)

    return np.column_stack([u, v])


def deduplicate_uvs(uv, vertex_texture, vertex_texture_map):
    vt_indices = []

    for u, v in uv:
        key = (round(float(u), 6), round(float(v), 6))

        if key not in vertex_texture_map:
            vertex_texture_map[key] = len(vertex_texture)
            vertex_texture.append([float(u), float(v)])

        vt_indices.append(vertex_texture_map[key])

    return vt_indices


# ============================================================
# APPEARANCE INITIALIZATION
# ============================================================
def init_appearance(texture_path):
    return {
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


def init_geom_appearance():
    return {
        "texture": {"default": {"values": []}},
        "material": {"default": {"values": []}}
    }

def process_face(
    face,
    face_type,
    has_semantics,
    vertices,
    transform,
    w,
    h,
    vertex_texture,
    vertex_texture_map,
    geom_appearance
):
    # Extract coordinates
    coords = get_face_coords(face, vertices)

    if coords.shape[0] < 3:
        return

    # Compute UVs
    uv = compute_uv(coords, transform, w, h)

    # Deduplicate UVs
    vt_indices = deduplicate_uvs(
        uv, vertex_texture, vertex_texture_map
    )

    # Decide if roof
    if has_semantics:
        roof = is_roof_semantic(face_type)
    else:
        roof = is_roof(coords)

    # Assign texture/material
    if roof:
        geom_appearance["texture"]["default"]["values"].append([vt_indices])
        geom_appearance["material"]["default"]["values"].append(0)
    else:
        geom_appearance["texture"]["default"]["values"].append(None)

        if not has_semantics:
            geom_appearance["material"]["default"]["values"].append(1)


# ============================================================
# MAIN PIPELINE
# ============================================================
def build_textured_cityjson(cityjson_path, tif_path, texture_path, out_path):

    cj, vertices = load_cityjson(cityjson_path)

    src = rasterio.open(tif_path)
    transform = src.transform
    w, h = src.width, src.height

    appearance = init_appearance(texture_path)

    vertex_texture = []
    vertex_texture_map = {}

    for obj in cj["CityObjects"].values():

        if obj["type"] != "Building":
            continue

        for geom in obj["geometry"]:

            geom["appearance"] = init_geom_appearance()

            faces = extract_faces(geom)
            surfaces = geom.get("semantics", {}).get("surfaces", [])
            has_semantics = len(surfaces) == len(faces)

            surface_types = [
                s.get("type", "WallSurface") for s in surfaces
            ] if has_semantics else [None] * len(faces)

            for face, face_type in zip(faces, surface_types):

                process_face(
                    face,
                    face_type,
                    has_semantics,
                    vertices,
                    transform,
                    w,
                    h,
                    vertex_texture,
                    vertex_texture_map,
                    geom["appearance"]
                )

    appearance["vertices-texture"] = vertex_texture
    cj["appearance"] = appearance

    with open(out_path, "w") as f:
        json.dump(cj, f)

    print("Saved:", out_path)


# ============================================================
# TEXTURE EXTRACTION
# ============================================================
def save_texture(tif_path, out_dir):
    with rasterio.open(tif_path) as src:
        img = src.read([1, 2, 3])
        img = np.transpose(img, (1, 2, 0))

        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.ptp() + 1e-12) * 255).astype(np.uint8)

    out_path = out_dir + "/texture.png"
    Image.fromarray(img).save(out_path)

    return Path(out_path).absolute().__str__()


# ============================================================
# RUN
# ============================================================
num = 5

CITYJSON_PATH = f"OK/roof{num}.json"
TIF_PATH = f"Orthos/roof{num}.tif"

OUT_DIR = Path("Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

texture_path = save_texture(TIF_PATH, "Outputs")

build_textured_cityjson(
    CITYJSON_PATH,
    TIF_PATH,
    texture_path,
    f"Outputs/roof{num}.json"
)