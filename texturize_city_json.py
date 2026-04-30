from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.transform import rowcol
from PIL import Image

# ============================================================
# STEP 1 — LOAD CITYJSON + APPLY GEOMETRY TRANSFORM
# (Equivalent: parsing GIS data into engine-ready vertex space)
# ============================================================
def load_cityjson(path):
    with open(path) as f:
        cj = json.load(f)

    # Extract global vertex array
    vertices = np.array(cj["vertices"])

    # If CityJSON uses compression transform, apply it
    if "transform" in cj:
        s = np.array(cj["transform"]["scale"])
        t = np.array(cj["transform"]["translate"])

        # Convert compressed coordinates → real-world coordinates
        vertices = vertices * s + t

    return cj, vertices



# ============================================================
# STEP 2 — DETECT ROOF SURFACES (GEOMETRY CLASSIFICATION)
# (Used to decide where textures will be applied)
# ============================================================
def is_roof(coords):
    # Build two edge vectors of the face
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]

    # Compute face normal via cross product
    n = np.cross(v1, v2)

    # Normalize vector (avoid division by zero)
    n = n / (np.linalg.norm(n) + 1e-12)

    # Roof heuristic: strong upward-facing normal (Z axis)
    return abs(n[2]) > 0.7



# ============================================================
# STEP 3 — COMPUTE UV COORDINATES FROM ORTHOPHOTO
# (Maps 3D coordinates → 2D texture space)
# ============================================================
def roof_uv(coords, transform, w, h):
    # Extract X/Y coordinates of polygon vertices
    xs = coords[:, 0]
    ys = coords[:, 1]

    # Convert world coordinates → raster pixel coordinates
    rows, cols = rowcol(transform, xs, ys)

    # Normalize pixel positions into UV space [0..1]
    u = np.array(cols) / w
    v = 1 - (np.array(rows) / h)

    # Return UV pairs per vertex
    return np.column_stack([u, v])



# ============================================================
# STEP 4 — MAIN PIPELINE: BUILD TEXTURED CITYJSON
# (Equivalent to: mesh + UV + material binding stage)
# ============================================================
def build_textured_cityjson(cityjson_path, tif_path, texture_path, out_path):

    # ----------------------------
    # STEP 4.1 — Load geometry + vertices
    # ----------------------------
    cj, vertices = load_cityjson(cityjson_path)

    # ----------------------------
    # STEP 4.2 — Load orthophoto metadata (for UV mapping)
    # ----------------------------
    src = rasterio.open(tif_path)
    transform = src.transform
    w, h = src.width, src.height

    # ============================================================
    # STEP 4.3 — CREATE GLOBAL APPEARANCE STRUCTURE
    # (Equivalent: defining materials + textures in render engine)
    # ============================================================
    appearance = {
        "textures": [
            {
                "type": "PNG",
                "image": texture_path   # texture atlas / orthophoto
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
        "vertices-texture": []  # global UV buffer (shared pool)
    }

    # UV deduplication system (avoid duplicate texture vertices)
    vertex_texture = []
    vertex_texture_map = {}

    # ============================================================
    # STEP 4.4 — ITERATE CITY OBJECTS (BUILDINGS ONLY)
    # (Equivalent: scene graph traversal)
    # ============================================================
    for obj_id, obj in cj["CityObjects"].items():

        if obj["type"] != "Building":
            continue

        # ========================================================
        # STEP 4.5 — ITERATE GEOMETRIES OF BUILDING
        # (Solid vs MultiSurface handling)
        # ========================================================
        for geom in obj["geometry"]:

            # Attach per-geometry appearance container
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

            # ----------------------------
            # STEP 4.6 — FLATTEN GEOMETRY INTO FACES
            # ----------------------------
            if geom["type"] == "Solid":
                # Solid = nested shells → flatten into face list
                all_faces = [f for shell in geom["boundaries"] for f in shell]
            else:
                all_faces = geom["boundaries"]

            # Semantic information (roof / wall classification if available)
            surfaces = geom.get('semantics', {}).get('surfaces', [])

            # ========================================================
            # STEP 4.7 — CASE 1: SEMANTICS MATCH FACE COUNT
            # (Use explicit roof/wall labels)
            # ========================================================
            if len(surfaces) == len(all_faces):

                surfaces_types = [s.get('type', 'WallSurface') for s in surfaces]

                for face, face_type in zip(all_faces, surfaces_types):

                    # Extract vertex indices of face
                    ring = face[0]
                    coords = np.array([vertices[i] for i in ring])

                    if coords.shape[0] < 3:
                        continue

                    # STEP 4.7.1 — Compute UV mapping
                    uv = roof_uv(coords, transform, w, h)

                    vt_indices = []

                    # STEP 4.7.2 — Deduplicate UV vertices globally
                    for u, v in uv:
                        key = (round(float(u), 6), round(float(v), 6))

                        if key not in vertex_texture_map:
                            vertex_texture_map[key] = len(vertex_texture)
                            vertex_texture.append([float(u), float(v)])

                        vt_indices.append(vertex_texture_map[key])

                    # ----------------------------
                    # ROOF SURFACE → APPLY TEXTURE
                    # ----------------------------
                    if face_type.lower().__contains__('roof'):
                        geom["appearance"]["texture"]["default"]["values"].append(
                            [vt_indices]
                        )
                        geom["appearance"]["material"]["default"]["values"].append(0)

                    # ----------------------------
                    # NON-ROOF → NO TEXTURE (fallback material)
                    # ----------------------------
                    else:
                        geom["appearance"]["texture"]["default"]["values"].append(None)

            # ========================================================
            # STEP 4.8 — CASE 2: NO SEMANTICS RELIABLY AVAILABLE
            # (Use geometric roof detection instead)
            # ========================================================
            else:
                for face in all_faces:

                    ring = face[0]
                    coords = np.array([vertices[i] for i in ring])

                    if coords.shape[0] < 3:
                        continue

                    # Compute UV mapping
                    uv = roof_uv(coords, transform, w, h)

                    vt_indices = []

                    # Deduplicate UVs
                    for u, v in uv:
                        key = (round(float(u), 6), round(float(v), 6))

                        if key not in vertex_texture_map:
                            vertex_texture_map[key] = len(vertex_texture)
                            vertex_texture.append([float(u), float(v)])

                        vt_indices.append(vertex_texture_map[key])

                    # ----------------------------
                    # ROOF DETECTED → APPLY TEXTURE
                    # ----------------------------
                    if is_roof(coords):
                        geom["appearance"]["texture"]["default"]["values"].append(
                            [vt_indices]
                        )
                        geom["appearance"]["material"]["default"]["values"].append(0)

                    # ----------------------------
                    # WALL → SOLID COLOR MATERIAL
                    # ----------------------------
                    else:
                        geom["appearance"]["texture"]["default"]["values"].append(None)
                        geom["appearance"]["material"]["default"]["values"].append(1)

    # ============================================================
    # STEP 5 — FINALIZE GLOBAL APPEARANCE BLOCK
    # (Attach UV buffer to CityJSON root)
    # ============================================================
    appearance["vertices-texture"] = vertex_texture
    cj["appearance"] = appearance

    # ============================================================
    # STEP 6 — EXPORT TEXTURED CITYJSON
    # (Final output consumed by viewer/render engine)
    # ============================================================
    with open(out_path, "w") as f:
        json.dump(cj, f)

    print("Saved:", out_path)



# ============================================================
# STEP 7 — EXTRACT ORTHOPHOTO INTO TEXTURE IMAGE
# (Creates PNG used by rendering pipeline)
# ============================================================
def save_texture(tif_path, out_dir):
    with rasterio.open(tif_path) as src:
        img = src.read([1,2,3])
        img = np.transpose(img, (1,2,0))

        # Normalize if needed
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.ptp() + 1e-12) * 255).astype(np.uint8)

    out_path = out_dir + "/texture.png"
    Image.fromarray(img).save(out_path)

    return Path(out_path).absolute().__str__()



# ============================================================
# STEP 8 — EXECUTION ENTRY POINT
# (Defines input dataset + runs full pipeline)
# ============================================================
num = 5

CITYJSON_PATH = f"OK/roof{num}.json"
TIF_PATH = f"Orthos/roof{num}.tif"
OUT_DIR = Path("Outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate texture image from orthophoto
texture_path = save_texture(TIF_PATH, 'Outputs')

# Build final textured CityJSON
build_textured_cityjson(
    CITYJSON_PATH,
    TIF_PATH,
    texture_path,
    f"Outputs/roof{num}.json"
)