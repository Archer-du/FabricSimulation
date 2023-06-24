# Author: Archer
import taichi as ti

ti.init(arch=ti.vulkan)

# GLOBAL CONFIG ===============================
# fabric arguments
n = 128
mass = 1.0
quad_size = 1.0 / n
x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

spring_offsets = []
for i in range(-2, 3):
    for j in range(-2, 3):
        if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
            spring_offsets.append(ti.Vector([i, j]))

# spring arguments
stiffness = 1e6
damping = 1e2

# collider arguments
ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0, 0, 0]
ball_stiffness = 1e5

# environment arguments
gravity = ti.Vector([0, -9.8, 0])
drag_damping = 1

# system arguments
dt = 4e-2 / n
substeps = int(1 / 60 // dt)
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

@ti.dataclass
class MassPoint:
    def __init__(self, mass: ti.f32 = 1.0):
        self.mass = mass
        self.position = ti.Vector([0, 0, 0])
        self.velocity = ti.Vector([0, 0, 0])

@ti.dataclass
class Spring:
    def __init__(self):
        self.stiffness = 0

@ti.dataclass
class Fabric:
    def __init__(self, massNum: ti.i32 = 128, length: ti.f32 = 1.0):
        self.massNum = massNum
        self.massPoints = ti.field(dtype=MassPoint, shape=(massNum, massNum))
        self.length = length
        self.quadSize = length / (massNum - 1)
        self.offsets = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                    self.offsets.append(ti.Vector([i, j]))
        #self.position = ti.Vector.field(3, dtype=ti.f32, shape=(massNum, massNum))
        #self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=(massNum, massNum))

@ti.dataclass
class Collider:
    def __init__(self, radius: ti.f32 = 0.3, center: ti.Vector = [0, 0, 0]):
        self.radius = radius
        self.center = ti.Vector.field(3, dtype=float, shape=(1, ))
        self.center[0] = center


@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    #random_offset = ti.Vector([0, 0])

    for i, j in x:
        x[i, j] = [
            i * quad_size - 0.5 + random_offset[0], 0.6,
            j * quad_size - 0.5 + random_offset[1]
        ]
        v[i, j] = [0, 0, 0]
        

# RENDER ======================================
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)

initialize_mesh_indices()

@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


# UPDATE ======================================
@ti.kernel
def substep():

    # environment adjustment
    for i in ti.grouped(x):
        # update velocity: gravity
        v[i] += gravity * dt
        # update velocity: air damping
        v[i] *= ti.exp(-drag_damping * dt)


    # Symplectic Euler analysis
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])

        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:

                # relative displacement
                x_ij = x[i] - x[j]
                # relative movement
                v_ij = v[i] - v[j]

                # vector normalization
                e_ij = x_ij.normalized()
                # modulus length
                cur_len = x_ij.norm()
                ori_len = quad_size * float(i - j).norm()

                # coefficient balance
                coe = (quad_size / ori_len)
                k_s = stiffness * coe
                k_d = damping

                # spring force
                force += -k_s * e_ij * (cur_len - ori_len)
                # spring damping
                force += -k_d * e_ij * v_ij.dot(e_ij)

        v[i] += (force / mass) * dt

    # collision detection
    for i in ti.grouped(x):
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal

        # update position
        x[i] += dt * v[i]




# GGUI settings
window = ti.ui.Window("Fabric Simulation", (2000, 2000))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize_mass_points()

while window.running:
    if current_t > 2.0:
        # Reset
        initialize_mass_points()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt
    update_vertices()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    # Draw a smaller ball to avoid visual penetration
    scene.particles(ball_center, radius=ball_radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()