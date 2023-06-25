import taichi as ti

ti.init(arch=ti.vulkan)

# 设置网格的大小和间隔
grid_size = 10
grid_step = 0.1

# 创建一个Taichi 3D向量场，用来存储网格的顶点位置
vertices = ti.Vector.field(3, dtype=ti.f32, shape=(grid_size + 1) ** 2)

# 遍历每个顶点，计算其位置
@ti.kernel
def init_vertices():
    for i, j in ti.ndrange(grid_size + 1, grid_size + 1):
        x = i * grid_step - 0.5
        y = j * grid_step - 0.5
        z = 0.0
        vertices[i * (grid_size + 1) + j] = [x, y, z]

# 调用init_vertices函数
init_vertices()

# 创建一个Taichi整数场，用来存储网格的线段索引
indices = ti.field(ti.i32, shape=grid_size * (grid_size + 1) * 2)

# 遍历每条线段，计算其索引
@ti.kernel
def init_indices():
    # 网格的水平线段
    for i, j in ti.ndrange(grid_size + 1, grid_size):
        k = i * grid_size + j
        indices[k * 2] = i * (grid_size + 1) + j
        indices[k * 2 + 1] = i * (grid_size + 1) + j + 1

    # 网格的垂直线段
    for i, j in ti.ndrange(grid_size, grid_size + 1):
        k = (grid_size + 1) * grid_size + i * grid_size + j
        indices[k * 2] = i * (grid_size + 1) + j
        indices[k * 2 + 1] = (i + 1) * (grid_size + 1) + j

# 调用init_indices函数
init_indices()

window = ti.ui.Window("Fabric Simulation", (1600, 1600))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.0, 0.0, 3)
camera.lookat(0.0, 0.0, 0)
scene.set_camera(camera)


while window.running:
    # 在scene中声明线段，用白色表示网格
    scene.lines(vertices, width=2.0, color=(1.0, 1.0, 1.0))

    # 在scene中声明一个红色的点，用来表示原点
    center = ti.Vector.field(3, dtype=float, shape=(1, ))
    center[0] = ti.Vector([0, 0, 0])
    scene.particles(center, radius=5.0, color=(1.0, 0.0, 0.0))
    window.show()
