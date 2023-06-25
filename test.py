import taichi as ti

ti.init(arch=ti.gpu)

# 创建一个窗口和一个scene
window = ti.ui.Window("Draw Line", res=(800, 600))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.0, 0.0, 3)
camera.lookat(0.0, 0.0, 0)
scene.set_camera(camera)

# 创建一个camera和一个light
camera = ti.ui.Camera()
camera.position(0.5, 1.5, 1.5)
camera.lookat(0.5, 0.5, 0.5)
light = scene.point_light(pos=(1.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))

# 创建一个3D向量场，用于存储线段的顶点
vertices = ti.Vector.field(3, dtype=ti.f32, shape=4)

# 初始化线段的顶点
@ti.kernel
def init():
    vertices[0] = [0.2, 0.2, 0.2]
    vertices[1] = [0.8, 0.2, 0.2]
    vertices[2] = [0.8, 0.8, 0.8]
    vertices[3] = [0.2, 0.8, 0.8]

init()

indices = ti.field(dtype=ti.i32, shape=(2 * 6))
indices[0] = 0 # 第一条线段的第一个顶点的索引
indices[1] = 1 # 第一条线段的第二个顶点的索引
indices[2] = 1 # 第二条线段的第一个顶点的索引
indices[3] = 2 # 第二条线段的第二个顶点的索引
indices[4] = 2 # 第三条线段的第一个顶点的索引
indices[5] = 3 # 第三条线段的第二个顶点的索引
indices[6] = 3 # 第四条线段的第一个顶点的索引
indices[7] = 0 # 第四条线段的第二个顶点的索引
indices[8] = 0 # 第五条线段的第一个顶点的索引
indices[9] = 2 # 第五条线段的第二个顶点的索引
indices[10] = 1 # 第六条线段的第一个顶点的索引
indices[11] = 3 # 第六条线段的第二个顶点的索引

# 绘制线段
while window.running:
    # 在scene上绘制两条线段，分别连接顶点0和1，以及顶点2和3
    scene.lines(vertices, width=5, indices=indices, color=(1.0, 0.0, 0.0))
    # 设置camera和light到scene
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    # 在scene上渲染并显示到窗口上
    canvas.scene(scene)
    window.show()