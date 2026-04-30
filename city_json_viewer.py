import json
import numpy as np
import pyvista as pv


# ----------------------------
# Triangulation
# ----------------------------
def triangulate_face(v_idx, uv_idx=None):
    tris = []

    for i in range(1, len(v_idx) - 1):
        tri_v = [v_idx[0], v_idx[i], v_idx[i + 1]]

        if uv_idx is not None and len(uv_idx) > 0:
            tri_uv = [uv_idx[0], uv_idx[i], uv_idx[i + 1]]
        else:
            tri_uv = None

        tris.append((tri_v, tri_uv))

    return tris


# ----------------------------
# MAIN CONVERSION
# ----------------------------
def cityjson_to_pyvista(cityjson_path):

    with open(cityjson_path) as f:
        data = json.load(f)

    verts = np.array(data["vertices"])

    appearance = data.get("appearance", {})
    uv_coords = np.array(appearance.get("vertices-texture", []))

    textures = appearance.get("textures", [])
    texture_path = textures[0]["image"] if textures else None

    materials = appearance.get("materials", [])

    # ----------------------------
    # Buffers
    # ----------------------------
    roof_points, roof_faces, roof_uvs = [], [], []
    wall_points, wall_faces = [], []

    roof_offset = 0
    wall_offset = 0

    # ----------------------------
    # Iterate objects
    # ----------------------------
    for obj in data["CityObjects"].values():
        for geom in obj.get("geometry", []):

            # Safe access
            tex_values = (
                geom.get("appearance", {})
                    .get("texture", {})
                    .get("default", {})
                    .get("values", [])
            )

            mat_values = (
                geom.get("appearance", {})
                    .get("material", {})
                    .get("default", {})
                    .get("values", [])
            )

            boundaries = geom.get("boundaries", [])

            for face_id, face in enumerate(boundaries):

                # -------- SAFE ACCESS --------
                tex_entry = tex_values[face_id] if face_id < len(tex_values) else []
                mat_id = mat_values[face_id] if face_id < len(mat_values) else 1

                ring = face[0]
                coords = verts[ring]

                # ----------------------------
                # ROOF (TEXTURED)
                # ----------------------------
                if tex_entry and len(tex_entry) > 0 and len(uv_coords) > 0:

                    uv_face = tex_entry[0]

                    tris = triangulate_face(ring, uv_face)

                    for tri_v_idx, tri_uv_idx in tris:

                        tri_points = verts[tri_v_idx]
                        roof_points.extend(tri_points)

                        roof_faces.extend([
                            3,
                            roof_offset,
                            roof_offset + 1,
                            roof_offset + 2
                        ])

                        for ui in tri_uv_idx:
                            roof_uvs.append(uv_coords[ui])

                        roof_offset += 3

                # ----------------------------
                # WALL (MATERIAL ONLY)
                # ----------------------------
                else:
                    tris = triangulate_face(ring, None)

                    for tri_v_idx, _ in tris:

                        tri_points = verts[tri_v_idx]
                        wall_points.extend(tri_points)

                        wall_faces.extend([
                            3,
                            wall_offset,
                            wall_offset + 1,
                            wall_offset + 2
                        ])

                        wall_offset += 3

    # ----------------------------
    # Build meshes
    # ----------------------------
    roof_mesh = None
    wall_mesh = None

    if len(roof_points) > 0:
        roof_mesh = pv.PolyData(
            np.array(roof_points),
            np.array(roof_faces)
        )
        roof_mesh.active_texture_coordinates = np.array(roof_uvs)

    if len(wall_points) > 0:
        wall_mesh = pv.PolyData(
            np.array(wall_points),
            np.array(wall_faces)
        )

    texture = pv.read_texture(texture_path) if texture_path else None

    return roof_mesh, wall_mesh, texture, materials


# ----------------------------
# VIEWER
# ----------------------------
def view(cityjson_path):

    roof_mesh, wall_mesh, texture, materials = cityjson_to_pyvista(cityjson_path)

    plotter = pv.Plotter()

    # ----------------------------
    # Roof
    # ----------------------------
    if roof_mesh is not None and texture is not None:
        plotter.add_mesh(roof_mesh, texture=texture)

    # ----------------------------
    # Walls
    # ----------------------------
    if wall_mesh is not None:

        wall_color = [0.6, 0.6, 0.6]  # default

        # try to read material
        if len(materials) > 1:
            mat = materials[1]
            if "diffuseColor" in mat:
                wall_color = mat["diffuseColor"]

        plotter.add_mesh(wall_mesh, color=wall_color)

    plotter.show()


# ----------------------------
# RUN
# ----------------------------
view("./Outputs/roof5.json")