import os
import cv2
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
from utils import dot, safe_normalize, cross
import time
import argparse
import pdb

class Mesh:
    """
    A torch-native trimesh class, with support for ``ply/obj/glb`` formats.

    Note:
        This class only supports one mesh with a single texture image.
    """
    def __init__(
        self,
        v=None,
        f=None,
        vn=None,
        fn=None,
        vt=None,
        ft=None,
        v_tangent=None,
        albedo=None,
        tactile_normal=None,
        metallicRoughness=None,
        vc=None, # vertex color
        device=None,
        opt=None,
        texture_map_size=1024,
    ):
        """Init a mesh directly using all attributes.

        Args:
            v (Optional[Tensor]): vertices, float [N, 3]. Defaults to None.
            f (Optional[Tensor]): faces, int [M, 3]. Defaults to None.
            vn (Optional[Tensor]): vertex normals, float [N, 3]. Defaults to None.
            fn (Optional[Tensor]): faces for normals, int [M, 3]. Defaults to None.
            vt (Optional[Tensor]): vertex uv coordinates, float [N, 2]. Defaults to None.
            ft (Optional[Tensor]): faces for uvs, int [M, 3]. Defaults to None.
            v_tangent (Optional[Tensor]): vertex tangent space, float [N, 3]. Defaults to None.
            albedo (Optional[Tensor]): albedo texture, float [H, W, 3], RGB format. Defaults to None.
            tactile_normal (Optional[Tensor]): tactile normal map, float [H, W, 3]. Defaults to None.
            metallicRoughness (Optional[Tensor]): metallic-roughness texture, float [H, W, 3], metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]. Defaults to None.
            vc (Optional[Tensor]): vertex colors, float [N, 3]. Defaults to None.
            device (Optional[torch.device]): torch device. Defaults to None.
            opt (Optional[argparse.Namespace]): options. Defaults to None.
            texture_map_size (int, optional): size of the texture map. Defaults to 1024.
        """        
        self.device = device
        self.v = v # vertices
        self.vn = vn # vertex normals
        self.vt = vt # vertex uv
        self.f = f # faces
        self.fn = fn # face normals 
        self.ft = ft # face uv
        self.v_tangent = v_tangent # vertex tangent space
        self.albedo = albedo
        self.tactile_normal = tactile_normal
        self.vc = vc # support vertex color if no albedo

        # pbr extension, metallic(Blue) = metallicRoughness[..., 2], roughness(Green) = metallicRoughness[..., 1]
        # ref: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
        # specifically Sec. 3.9 https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials
        self.metallicRoughness = metallicRoughness
        self.ori_center = 0
        self.ori_scale = 1
        self.opt = opt # inherit the opt from the renderer
        self.texture_map_size = texture_map_size
        # initialize an empty label map 
        self.label_map = torch.zeros((texture_map_size, texture_map_size, 3), dtype=torch.float32, device=device)
    
    @classmethod
    def load(cls, path, resize=True, renormal=True, retex=False, bound=0.9, front_dir='+z', opt=None, **kwargs):
        """load mesh from path.

        Args:
            path (str): path to mesh file, supports ply, obj, glb.
            resize (bool, optional): auto resize the mesh using ``bound`` into [-bound, bound]^3. Defaults to True.
            renormal (bool, optional): re-calc the vertex normals. Defaults to True.
            retex (bool, optional): re-calc the uv coordinates, will overwrite the existing uv coordinates. Defaults to False.
            bound (float, optional): bound to resize. Defaults to 0.9.
            front_dir (str, optional): front-view direction of the mesh, should be [+-][xyz][ 123]. Defaults to '+z'.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note:
            a ``device`` keyword argument can be provided to specify the torch device. 
            If it's not provided, we will try to use ``'cuda'`` as the device if it's available.

        Returns:
            Mesh: the loaded Mesh object.
        """

        # obj supports face uv
        if path.endswith(".obj"):
            mesh = cls.load_obj(path, **kwargs, opt=opt)
        # trimesh only supports vertex uv, but can load more formats
        else:
            mesh = cls.load_trimesh(path, **kwargs, opt=opt)

        # Load options from the renderer
        if opt is not None:
            if mesh.opt is None:
                mesh.opt = opt
            else:
                mesh.opt.update(opt)

        print(f"[Mesh loading] v: {mesh.v.shape}, f: {mesh.f.shape}")
        # auto-normalize
        if resize:
            print(f"auto_size ...")
            start_time = time.time()
            mesh.auto_size(bound=bound)
            print(f"[INFO] load mesh, auto_size: {time.time() - start_time:.4f}s")
        # auto-fix normal
        if renormal or mesh.vn is None:
            print(f"auto_normal ...")
            start_time = time.time()
            mesh.auto_normal()
            print(f"[INFO] load mesh, auto_normal: {time.time() - start_time:.4f}s")
            print(f"[INFO] load mesh, vn: {mesh.vn.shape}, fn: {mesh.fn.shape}") # vn: torch.Size([118570, 3]), fn: torch.Size([237140, 3])
        # auto-fix texcoords
        if retex or (mesh.vt is None and mesh.albedo is not None) or (mesh.vt is None and mesh.tactile_normal is not None):
            print(f"auto_uv ...")
            start_time = time.time()
            mesh.auto_uv(cache_path=path)
            print(f"[INFO] load mesh, auto_uv: {time.time() - start_time:.4f}s")

        # rotate front dir to +z
        if front_dir != "+z":
            # axis switch
            if "-z" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], device=mesh.device, dtype=torch.float32)
            elif "+x" in front_dir:
                T = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "-x" in front_dir:
                T = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=mesh.device, dtype=torch.float32)
            elif "+y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            elif "-y" in front_dir:
                T = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], device=mesh.device, dtype=torch.float32)
            else:
                T = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32)
            # rotation (how many 90 degrees)
            if '1' in front_dir:
                T @= torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '2' in front_dir:
                T @= torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            elif '3' in front_dir:
                T @= torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=mesh.device, dtype=torch.float32) 
            mesh.v @= T
            mesh.vn @= T

        return mesh

    @classmethod
    def load_tactile_texture(cls, tactile_normal_path, texture_crop_ratio=1.0):
        """
        Sample code to load tactile texture from a separate file.
        Create a function here so that we don't need to duplicate for load_obj and load_trimesh functions.
        """
        # Load the 3-channel normal map
        print(f"Load tactile normal from {tactile_normal_path}")
        tactile_normal = cv2.imread(tactile_normal_path, cv2.IMREAD_UNCHANGED)
        tactile_normal = cv2.cvtColor(tactile_normal, cv2.COLOR_BGR2RGB)
        tactile_normal = tactile_normal.astype(np.float32) / 255.0 # shape [1024, 1024, 3], range [0, 1]
        # convert range [0, 1] to [-1, 1]
        tactile_normal = tactile_normal * 2.0 - 1.0 
        tactile_normal_norm = np.linalg.norm(tactile_normal, axis=-1)

        # if opt.texture_crop_ratio is provided, crop the texture (more flexibility to match the texture with mesh's physical scale)
        crop_size = int(tactile_normal.shape[0] * texture_crop_ratio)
        print(f"center crop tactile texture from {tactile_normal.shape} to size ({crop_size}, {crop_size}), texture_crop_ratio = {texture_crop_ratio}")
        start_h = (tactile_normal.shape[0] - crop_size) // 2
        start_w = (tactile_normal.shape[1] - crop_size) // 2
        tactile_normal = tactile_normal[start_h:start_h+crop_size, start_w:start_w+crop_size]
        # resize to nearest number of power of 2 so that the mipmapping works
        new_size = 2 ** int(np.log2(crop_size))
        tactile_normal = cv2.resize(tactile_normal, (new_size, new_size), interpolation=cv2.INTER_LANCZOS4)
        print(f"resize tactile texture to {tactile_normal.shape}")
        return tactile_normal

    @classmethod
    def load_label_map(cls, label_map_path, texture_crop_ratio=1.0):
        """
        Load part label map from a separate file.
        """
        label_map = cv2.imread(label_map_path, cv2.IMREAD_UNCHANGED)
        label_map = cv2.cvtColor(label_map, cv2.COLOR_BGR2RGB)
        # R(255, 0, 0) - partA; G(0, 255, 0) - partB
        # convert to float and divide by 255
        label_map = label_map.astype(np.float32) / 255.0 # R(1.0, 0, 0) - partA; G(0, 1.0, 0) - partB
        # center crop
        crop_size = int(label_map.shape[0] * texture_crop_ratio)
        start_h = (label_map.shape[0] - crop_size) // 2
        start_w = (label_map.shape[1] - crop_size) // 2
        label_map = label_map[start_h:start_h+crop_size, start_w:start_w+crop_size]
        # resize to nearest number of power of 2 so that the mipmapping works
        new_size = 2 ** int(np.log2(crop_size))
        label_map = cv2.resize(label_map, (new_size, new_size), interpolation=cv2.INTER_LANCZOS4)
        print(f"resize label_map to {label_map.shape}")
        return label_map

    
    # load from obj file
    @classmethod
    def load_obj(cls, path, albedo_path=None, device=None, opt=None):
        """load an ``obj`` mesh.

        Args:
            path (str): path to mesh.
            albedo_path (str, optional): path to the albedo texture image, will overwrite the existing texture path if specified in mtl. Defaults to None.
            device (torch.device, optional): torch device. Defaults to None.
        
        Note: 
            We will try to read `mtl` path from `obj`, else we assume the file name is the same as `obj` but with `mtl` extension.
            The `usemtl` statement is ignored, and we only use the last material path in `mtl` file.

        Returns:
            Mesh: the loaded Mesh object.
        """

        print(f"load mesh using load_obj, path: {path}")
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mesh.device = device

        # load obj
        with open(path, "r") as f:
            lines = f.readlines()

        def parse_f_v(fv):
            # pass in a vertex term of a face, return {v, vt, vn} (-1 if not provided)
            # supported forms:
            # f v1 v2 v3
            # f v1/vt1 v2/vt2 v3/vt3
            # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
            # f v1//vn1 v2//vn2 v3//vn3
            xs = [int(x) - 1 if x != "" else -1 for x in fv.split("/")]
            xs.extend([-1] * (3 - len(xs)))
            return xs[0], xs[1], xs[2]

        # NOTE: we ignore usemtl, and assume the mesh ONLY uses one material (first in mtl)
        vertices, texcoords, normals = [], [], []
        faces, tfaces, nfaces = [], [], []
        mtl_path = None

        for line in lines:
            split_line = line.split()
            # empty line
            if len(split_line) == 0:
                continue
            prefix = split_line[0].lower()
            # mtllib
            if prefix == "mtllib":
                mtl_path = split_line[1]
            # usemtl
            elif prefix == "usemtl":
                pass # ignored
            # v/vn/vt
            elif prefix == "v":
                vertices.append([float(v) for v in split_line[1:]])
            elif prefix == "vn":
                normals.append([float(v) for v in split_line[1:]])
            elif prefix == "vt":
                val = [float(v) for v in split_line[1:]]
                texcoords.append([val[0], 1.0 - val[1]])
            elif prefix == "f":
                vs = split_line[1:]
                nv = len(vs)
                v0, t0, n0 = parse_f_v(vs[0])
                for i in range(nv - 2):  # triangulate (assume vertices are ordered)
                    v1, t1, n1 = parse_f_v(vs[i + 1])
                    v2, t2, n2 = parse_f_v(vs[i + 2])
                    faces.append([v0, v1, v2])
                    tfaces.append([t0, t1, t2])
                    nfaces.append([n0, n1, n2])

        mesh.v = torch.tensor(vertices, dtype=torch.float32, device=device)
        # print per axis min and max
        print(f"Check mesh's size while loading: {mesh.v.shape}")
        for i in range(3):
            print(f"axis {i}: min {mesh.v[:, i].min()}, max {mesh.v[:, i].max()}")

        mesh.vt = (
            torch.tensor(texcoords, dtype=torch.float32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.vn = (
            torch.tensor(normals, dtype=torch.float32, device=device)
            if len(normals) > 0
            else None
        )

        mesh.f = torch.tensor(faces, dtype=torch.int32, device=device)
        mesh.ft = (
            torch.tensor(tfaces, dtype=torch.int32, device=device)
            if len(texcoords) > 0
            else None
        )
        mesh.fn = (
            torch.tensor(nfaces, dtype=torch.int32, device=device)
            if len(normals) > 0
            else None
        )

        # see if there is vertex color
        use_vertex_color = False
        if mesh.v.shape[1] == 6:
            use_vertex_color = True # so that we skip the next section of loading albedo from image.
            mesh.vc = mesh.v[:, 3:]
            mesh.v = mesh.v[:, :3]
            print(f"[INFO] load obj mesh: use vertex color: {mesh.vc.shape}")


        # try to load texture image
        if not use_vertex_color:
            # try to retrieve mtl file
            mtl_path_candidates = []
            if mtl_path is not None:
                mtl_path_candidates.append(mtl_path)
                mtl_path_candidates.append(os.path.join(os.path.dirname(path), mtl_path))
            mtl_path_candidates.append(path.replace(".obj", ".mtl"))

            mtl_path = None
            for candidate in mtl_path_candidates:
                if os.path.exists(candidate):
                    mtl_path = candidate
                    break
            
            # if albedo_path is not provided, try retrieve it from mtl
            metallic_path = None
            roughness_path = None
            if mtl_path is not None and albedo_path is None:
                with open(mtl_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    split_line = line.split()
                    # empty line
                    if len(split_line) == 0:
                        continue
                    prefix = split_line[0]
                    
                    if "map_Kd" in prefix:
                        # assume relative path!
                        albedo_path = os.path.join(os.path.dirname(path), split_line[1])
                        print(f"[INFO] load obj mesh: use texture from: {albedo_path}")
                    elif "map_Pm" in prefix:
                        metallic_path = os.path.join(os.path.dirname(path), split_line[1])
                    elif "map_Pr" in prefix:
                        roughness_path = os.path.join(os.path.dirname(path), split_line[1])
                    
            # still not found albedo_path, or the path doesn't exist
            if albedo_path is None or not os.path.exists(albedo_path):
                # init an empty texture
                print(f"[load_obj] init empty albedo!")
                # albedo = np.random.rand(1024, 1024, 3).astype(np.float32)
                albedo = np.ones((1024, 1024, 3), dtype=np.float32) * np.array([0.5, 0.5, 0.5])  # default color
            else:
                albedo = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
                albedo = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
                albedo = albedo.astype(np.float32) / 255
                print(f"[load_obj] load albedo texture: {albedo.shape}, range [{albedo.min()}, {albedo.max()}]") # shape (1024, 1024, 3)

            mesh.albedo = torch.tensor(albedo, dtype=torch.float32, device=device)


            # try to load metallic and roughness
            if metallic_path is not None and roughness_path is not None:
                print(f"[INFO] load obj mesh: load metallicRoughness from: {metallic_path}, {roughness_path}")
                metallic = cv2.imread(metallic_path, cv2.IMREAD_UNCHANGED)
                metallic = metallic.astype(np.float32) / 255
                roughness = cv2.imread(roughness_path, cv2.IMREAD_UNCHANGED)
                roughness = roughness.astype(np.float32) / 255
                metallicRoughness = np.stack([np.zeros_like(metallic), roughness, metallic], axis=-1)

                mesh.metallicRoughness = torch.tensor(metallicRoughness, dtype=torch.float32, device=device).contiguous()
        
        # Load tactile texture from a separate file. Output: torch tensor of shape [1024, 1024, 3], range [0, 1], dtype float32
        if opt is None:
            no_tactile = True
        else:
            if hasattr(opt, "no_tactile") and opt.no_tactile:
                no_tactile = True
            else:
                if hasattr(opt, "load_tactile"):
                    no_tactile = not opt.load_tactile
                else:
                    no_tactile = False
        
        if not no_tactile:
            # Load tactile normal to mesh  object

            if opt is None or not hasattr(opt, "tactile_normal_path"):
                # set default tactile texture path
                tactile_normal_path = path.replace(".obj", "_tactile_normal.png")
                if opt is None:
                    opt = argparse.Namespace(tactile_normal_path=tactile_normal_path)
                if not hasattr(opt, "tactile_normal_path"):
                    opt.tactile_normal_path = tactile_normal_path

                assert os.path.exists(opt.tactile_normal_path), f"tactile_normal_path {opt.tactile_normal_path} not found!"

            texture_crop_ratio = opt.texture_crop_ratio if hasattr(opt, "texture_crop_ratio") else 1.0
            tactile_normal = cls.load_tactile_texture(tactile_normal_path=opt.tactile_normal_path, texture_crop_ratio=texture_crop_ratio)
            mesh.tactile_normal = torch.tensor(tactile_normal, dtype=torch.float32, device=device)
            print(f"[load_obj] load tactile normal: type {type(mesh.tactile_normal)} shape {mesh.tactile_normal.shape}, range min {mesh.tactile_normal.min()}, max {mesh.tactile_normal.max()}, dtype {mesh.tactile_normal.dtype}") # type <class 'torch.Tensor'> shape torch.Size([1024, 1024, 3]), range min -1.0, max 1.0, dtype torch.float32 # the perturbation should be in the range of [-1, 1]

            if hasattr(opt, "num_part_label") and opt.num_part_label > 0 :
                print(f"we have num_part_label {opt.num_part_label}, load the second tactile normal map.")
                # multi-parts
                # load second tactile normal
                tactile_normal2 = cls.load_tactile_texture(tactile_normal_path=opt.tactile_normal_path2, texture_crop_ratio=texture_crop_ratio)
                mesh.tactile_normal2 = torch.tensor(tactile_normal2, dtype=torch.float32, device=device)
                print(f"[load_obj] load tactile normal2: type {type(mesh.tactile_normal2)} shape {mesh.tactile_normal2.shape}, range min {mesh.tactile_normal2.min()}, max {mesh.tactile_normal2.max()}, dtype {mesh.tactile_normal2.dtype}") 

            # Load part label map, only if multi-part tactile normal is loaded and the lable map path exists
            label_map_path = opt.tactile_normal_path.replace("_tactile_normal.png", "_label_map.png")
            if os.path.exists(label_map_path):
                print(f"Load part label map from {label_map_path}")
            label_map = cls.load_label_map(label_map_path, texture_crop_ratio=texture_crop_ratio) # shape [1024, 1024, 3], range [0, 1], dtype float32
            mesh.label_map = torch.tensor(label_map, dtype=torch.float32, device=device)
            print(f"[load_obj] load label map: type {type(mesh.label_map)} shape {mesh.label_map.shape}, range min {mesh.label_map.min()}, max {mesh.label_map.max()}, dtype {mesh.label_map.dtype}")
            
        else:
            print(f"[load_obj] no tactile texture loaded.")

        return mesh

    @classmethod
    def load_trimesh(cls, path, device=None):
        """load a mesh using ``trimesh.load()``.

        Can load various formats like ``glb`` and serves as a fallback.

        Note:
            We will try to merge all meshes if the glb contains more than one, 
            but **this may cause the texture to lose**, since we only support one texture image!

        Args:
            path (str): path to the mesh file.
            device (torch.device, optional): torch device. Defaults to None.

        Returns:
            Mesh: the loaded Mesh object.
        """
        print(f"load mesh using load_obj")
        raise NotImplementedError("load_trimesh is not implemented fully yet for tactile loading !")

  
    # sample surface (using trimesh)
    def sample_surface(self, count: int):
        """sample points on the surface of the mesh.

        Args:
            count (int): number of points to sample.

        Returns:
            torch.Tensor: the sampled points, float [count, 3].
        """
        _mesh = trimesh.Trimesh(vertices=self.v.detach().cpu().numpy(), faces=self.f.detach().cpu().numpy())
        points, face_idx = trimesh.sample.sample_surface(_mesh, count)
        points = torch.from_numpy(points).float().to(self.device)
        return points

    # aabb
    def aabb(self):
        """get the axis-aligned bounding box of the mesh.

        Returns:
            Tuple[torch.Tensor]: the min xyz and max xyz of the mesh.
        """
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self, bound=0.9):
        """auto resize the mesh.

        Args:
            bound (float, optional): resizing into ``[-bound, bound]^3``. Defaults to 0.9.
        """
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = 2 * bound / torch.max(vmax - vmin).item()
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal(self):
        """auto calculate the vertex normals.
        """
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

        # compute tangent space so that it is updated every time the normal is updated
        start_time = time.time()
        # auto_tangent requires ft, so we need to check if it's available
        if self.ft is None:
            # run auto_uv to generate texture coordinates
            print(f"[INFO], self.ft is None, run auto_uv to generate texture coordinates")
            self.auto_uv(vmap=True) # vmap runs auto_tangent
        else:
            self.auto_tangent()
        print(f"[INFO] auto_tangent: {time.time() - start_time:.4f}s")


    def auto_tangent(self):
        """
        Compute per-vertex tangent space basis.
        Ref: https://github.com/threestudio-project/threestudio/blob/cd462fb0b73a89b6be17160f7802925fe6cf34cd/threestudio/models/mesh.py#L162
        """
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(3):
            pos[i] = self.v[self.f[:, i]]
            tex[i] = self.vt[self.ft[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.f[:, i]
        
        tangents = torch.zeros_like(self.vn)
        bitangents = torch.zeros_like(self.vn)
        tansum = torch.zeros_like(self.vn)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        nom_bitan =  pe2 * uve1[..., 0:1] - pe1 * uve2[..., 0:1]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )
        bitang = nom_bitan / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3).type(torch.int64)
        
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            bitangents.scatter_add_(0, idx, bitang)  # bitangents[n_i] = bitangents[n_i] + bitang
  
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1

        tangents = tangents / tansum
        bitangents = bitangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.vn) * self.vn)
        handedness = torch.sign(dot(cross(self.vn, tangents), bitangents))

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        self.v_tangent = tangents
        

    def auto_uv(self, cache_path=None, vmap=True):
        """auto calculate the uv coordinates.

        Args:
            cache_path (str, optional): path to save/load the uv cache as a npz file, this can avoid calculating uv every time when loading the same mesh, which is time-consuming. Defaults to None.
            vmap (bool, optional): remap vertices based on uv coordinates, so each v correspond to a unique vt (necessary for formats like gltf). 
                Usually this will duplicate the vertices on the edge of uv atlas. Defaults to True.
        """
        # try to load cache
        if cache_path is not None:
            cache_path = os.path.splitext(cache_path)[0] + "_uv.npz"
        if cache_path is not None and os.path.exists(cache_path):
            data = np.load(cache_path)
            vt_np, ft_np, vmapping = data["vt"], data["ft"], data["vmapping"]
        else:
            import xatlas

            v_np = self.v.detach().cpu().numpy()
            f_np = self.f.detach().int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            # chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            # save to cache
            if cache_path is not None:
                np.savez(cache_path, vt=vt_np, ft=ft_np, vmapping=vmapping)
        
        vt = torch.from_numpy(vt_np.astype(np.float32)).to(self.device)
        ft = torch.from_numpy(ft_np.astype(np.int32)).to(self.device)
        self.vt = vt
        self.ft = ft

        if vmap:
            # remap v/f to vt/ft, so each v correspond to a unique vt. (necessary for gltf)
            vmapping = torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
            self.align_v_to_vt(vmapping)
    

    def align_v_to_vt(self, vmapping=None):
        """ remap v/f and vn/fn to vt/ft.

        Args:
            vmapping (np.ndarray, optional): the mapping relationship from f to ft. Defaults to None.
        """
        if vmapping is None:
            ft = self.ft.view(-1).long()
            f = self.f.view(-1).long()
            vmapping = torch.zeros(self.vt.shape[0], dtype=torch.long, device=self.device)
            vmapping[ft] = f # scatter, randomly choose one if index is not unique
        self.v = self.v[vmapping]
        self.f = self.ft
        # assume fn == f
        if self.vn is not None:
            self.vn = self.vn[vmapping]
            # since we update vn, we need to update tangent space too
            self.auto_tangent()
            self.fn = self.ft
        if self.vc is not None:
            self.vc = self.vc[vmapping]

    def to(self, device):
        """move all tensor attributes to device.

        Args:
            device (torch.device): target device.

        Returns:
            Mesh: self.
        """
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    

    def write(self, path):
        """write the mesh to a path.

        Args:
            path (str): path to write, supports ply, obj and glb.
        """
        assert path.endswith(".obj"), "Currently only suppport .obj format to save tactile texture"
        
        if path.endswith(".ply"):
            self.write_ply(path)
        elif path.endswith(".obj"):
            self.write_obj(path)
        else:
            raise NotImplementedError(f"format {path} not supported!")
    
    # write to ply file (only geom)
    def write_ply(self, path):
        """write the mesh in ply format. Only for geometry!

        Args:
            path (str): path to write.
        """

        if self.albedo is not None:
            print(f'[WARN] ply format does not support exporting texture, will ignore!')

        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        _mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
        _mesh.export(path)


    # write to obj file (geom + texture)
    def write_obj(self, path):
        """write the mesh in obj format. Will also write the texture and mtl files.

        Args:
            path (str): path to write.
        """

        mtl_path = path.replace(".obj", ".mtl")
        albedo_path = path.replace(".obj", "_albedo.png")
        tactile_normal_path = path.replace(".obj", "_tactile_normal.png")
        label_map_path = path.replace(".obj", "_label_map.png")

        metallic_path = path.replace(".obj", "_metallic.png")
        roughness_path = path.replace(".obj", "_roughness.png")

        v_np = self.v.detach().cpu().numpy()
        vt_np = self.vt.detach().cpu().numpy() if self.vt is not None else None
        vn_np = self.vn.detach().cpu().numpy() if self.vn is not None else None
        f_np = self.f.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy() if self.ft is not None else None
        fn_np = self.fn.detach().cpu().numpy() if self.fn is not None else None

        print(f"[INFO] write to {path} ... Updated N_v {v_np.shape[0]}, N_f {f_np.shape[0]}")

        with open(path, "w") as fp:
            fp.write(f"mtllib {os.path.basename(mtl_path)} \n")

            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            if vt_np is not None:
                for v in vt_np:
                    fp.write(f"vt {v[0]} {1 - v[1]} \n")

            if vn_np is not None:
                for v in vn_np:
                    fp.write(f"vn {v[0]} {v[1]} {v[2]} \n")

            fp.write(f"usemtl defaultMat \n")
            for i in range(len(f_np)):
                fp.write(
                    f'f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1 if ft_np is not None else ""}/{fn_np[i, 0] + 1 if fn_np is not None else ""} \
                             {f_np[i, 1] + 1}/{ft_np[i, 1] + 1 if ft_np is not None else ""}/{fn_np[i, 1] + 1 if fn_np is not None else ""} \
                             {f_np[i, 2] + 1}/{ft_np[i, 2] + 1 if ft_np is not None else ""}/{fn_np[i, 2] + 1 if fn_np is not None else ""} \n'
                )

        with open(mtl_path, "w") as fp:
            fp.write(f"newmtl defaultMat \n")
            fp.write(f"Ka 1 1 1 \n")
            fp.write(f"Kd 1 1 1 \n")
            fp.write(f"Ks 0 0 0 \n")
            fp.write(f"Tr 1 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0 \n")
            if self.albedo is not None:
                fp.write(f"map_Kd {os.path.basename(albedo_path)} \n")
            if self.metallicRoughness is not None:
                # ref: https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_Rendering
                fp.write(f"map_Pm {os.path.basename(metallic_path)} \n")
                fp.write(f"map_Pr {os.path.basename(roughness_path)} \n")

        if self.albedo is not None:
            albedo = self.albedo.detach().cpu().numpy()
            albedo = (albedo * 255).astype(np.uint8)
            cv2.imwrite(albedo_path, cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
        if self.tactile_normal is not None:
            print(f"Saving tactile_normal to {tactile_normal_path}")
            tactile_normal = self.tactile_normal.detach().cpu().numpy()
            print(f"Saving tactile_normal to {tactile_normal_path}, current range: [{tactile_normal.min()}, {tactile_normal.max()}], dtype {tactile_normal.dtype}, shape {tactile_normal.shape}")
            # normalize to unit length
            tactile_normal = tactile_normal / np.linalg.norm(tactile_normal, axis=-1, keepdims=True)
            print(f"after normalization, tactile_normal range: [{tactile_normal.min()}, {tactile_normal.max()}]")
            # convert tactile normal range from [-1, 1] to [0, 1] 
            tactile_normal = (tactile_normal + 1.0) / 2.0
            print(f"tactile_normal range after conversion: [{tactile_normal.min()}, {tactile_normal.max()}]")
            # convert to [0, 255] and save as uint8
            tactile_normal = (tactile_normal * 255).astype(np.uint8)
            cv2.imwrite(tactile_normal_path, cv2.cvtColor(tactile_normal, cv2.COLOR_RGB2BGR))
        if self.label_map is not None:
            # NOTE: now the label_map is already converted to 3 channels, range [0, 1] in the uv_padding function
            label_map = self.label_map.detach().cpu().numpy()
            label_map = (label_map * 255).astype(np.uint8)
            cv2.imwrite(label_map_path, cv2.cvtColor(label_map, cv2.COLOR_RGB2BGR))
            
        if self.metallicRoughness is not None:
            metallicRoughness = self.metallicRoughness.detach().cpu().numpy()
            metallicRoughness = (metallicRoughness * 255).astype(np.uint8)
            cv2.imwrite(metallic_path, metallicRoughness[..., 2])
            cv2.imwrite(roughness_path, metallicRoughness[..., 1])
