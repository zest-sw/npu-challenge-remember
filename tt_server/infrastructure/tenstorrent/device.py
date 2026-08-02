import ttnn


MESH_SHAPES = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}


class DeviceFactory:
    def __init__(self, mesh_name: str | None, trace_region_size: int):
        self.mesh_name = mesh_name
        self.trace_region_size = trace_region_size

    def open(self):
        device_ids = ttnn.get_device_ids()
        shape = MESH_SHAPES.get(self.mesh_name)
        if shape is None:
            shape = (1, min(len(device_ids), len(device_ids)))
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*shape),
            trace_region_size=self.trace_region_size,
            num_command_queues=2,
        )

    @staticmethod
    def synchronize(device) -> None:
        ttnn.synchronize_device(device)

    @staticmethod
    def close(device) -> None:
        ttnn.close_mesh_device(device)
