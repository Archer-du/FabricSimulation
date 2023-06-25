# Author: Archer
import taichi as ti

ti.init(arch=ti.vulkan)


# DATA CLASS ==================================
@ti.data_oriented
class Fabric:
    
    def __init__(self, mass: ti.f32 = 1.0, massNum: ti.i32 = 128, length: ti.f32 = 1.25):
        # simulation arguments
        self.mass = mass
        self.massNum = massNum
        self.length = length
        self.quadSize = length / (massNum - 1)
        self.offsets = []
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=(massNum, massNum))
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=(massNum, massNum))
        self.initSpringOffset()
        self.InitMassPoints()

        # render arguments
        self.triangleNum = (massNum - 1) * (massNum - 1) * 2
        self.gridNum = (massNum - 1) * (massNum - 1)
        self.triangleIndices = ti.field(ti.i32, shape=self.triangleNum * 3)
        self.gridIndices = ti.field(ti.i32, shape=(2 * self.gridNum * 6))
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=massNum * massNum)
        self.triangleColors = ti.Vector.field(3, dtype=ti.f32, shape=massNum * massNum)
        self.gridColors = ti.Vector.field(3, dtype=ti.f32, shape=(self.gridNum * 6))
        self.InitTriangleMeshIndices()
        self.InitGridMeshIndices()

    def initSpringOffset(self):
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                    self.offsets.append(ti.Vector([i, j]))

    # SIM INIT -------------
    @ti.kernel
    def InitMassPoints(self):
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1

        for i, j in self.position:
            self.position[i, j] = [
                i * self.quadSize - self.length / 2 + random_offset[0], 0.6,
                j * self.quadSize - self.length / 2 + random_offset[1]
            ]
            self.velocity[i, j] = [0, 0, 0]

    # SIM UPDATE -----------
    @ti.kernel
    def UpdateSys(self):
        # environment adjustment
        for i in ti.grouped(self.position):
            # update velocity: gravity
            self.velocity[i] += GRAVITY * dt
            # update velocity: air DAMPING
            self.velocity[i] *= ti.exp(-AIR_DRAG * dt)

        # Symplectic Euler analysis
        for i in ti.grouped(self.position):
            force = ti.Vector([0.0, 0.0, 0.0])

            for springOffset in ti.static(self.offsets):
                j = i + springOffset
                if 0 <= j[0] < self.massNum and 0 <= j[1] < self.massNum:

                    # relative displacement
                    x_ij = self.position[i] - self.position[j]
                    # relative movement
                    v_ij = self.velocity[i] - self.velocity[j]

                    # vector normalization
                    e_ij = x_ij.normalized()
                    # modulus length
                    cur_len = x_ij.norm()
                    ori_len = self.quadSize * float(i - j).norm()

                    # coefficient balance
                    coe = (self.quadSize / ori_len)
                    k_s = STIFFNESS * coe
                    k_d = DAMPING

                    # spring force
                    force += -k_s * e_ij * (cur_len - ori_len)
                    # spring damping
                    force += -k_d * e_ij * v_ij.dot(e_ij)

            self.velocity[i] += (force / self.mass) * dt

        # collision detection
        for i in ti.grouped(self.position):
            offset_to_center = self.position[i] - sphere.center[0]
            if offset_to_center.norm() <= sphere.radius:
                e = offset_to_center.normalized()
                self.velocity[i] -= min(self.velocity[i].dot(e), 0) * e

            # update position
            self.position[i] += dt * self.velocity[i]

    # RENDER INIT -------------
    @ti.kernel
    def InitTriangleMeshIndices(self):
        for i, j in ti.ndrange(self.massNum - 1, self.massNum - 1):
            quad_id = (i * (self.massNum - 1)) + j
            # 1st triangle of the square
            self.triangleIndices[quad_id * 6 + 0] = i * self.massNum + j
            self.triangleIndices[quad_id * 6 + 1] = (i + 1) * self.massNum + j
            self.triangleIndices[quad_id * 6 + 2] = i * self.massNum + (j + 1)
            # 2nd triangle of the square
            self.triangleIndices[quad_id * 6 + 3] = (i + 1) * self.massNum + (j + 1)
            self.triangleIndices[quad_id * 6 + 4] = i * self.massNum + (j + 1)
            self.triangleIndices[quad_id * 6 + 5] = (i + 1) * self.massNum + j

        for i, j in ti.ndrange(self.massNum, self.massNum):
                self.triangleColors[i * self.massNum + j] = (106/255, 90/255, 205/255)

    @ti.kernel
    def InitGridMeshIndices(self):
        for i, j in ti.ndrange(self.massNum - 1, self.massNum - 1):
            grid_id = (i * (self.massNum - 1)) + j
            self.gridIndices[grid_id * 12 + 0] = i * self.massNum + j
            self.gridIndices[grid_id * 12 + 1] = i * self.massNum + (j + 1)
            self.gridIndices[grid_id * 12 + 2] = i * self.massNum + j
            self.gridIndices[grid_id * 12 + 3] = (i + 1) * self.massNum + j
            self.gridIndices[grid_id * 12 + 4] = i * self.massNum + j
            self.gridIndices[grid_id * 12 + 5] = (i + 1) * self.massNum + (j + 1)
            self.gridIndices[grid_id * 12 + 6] = (i + 1) * self.massNum + j
            self.gridIndices[grid_id * 12 + 7] = i * self.massNum + (j + 1)
            self.gridIndices[grid_id * 12 + 8] = (i + 1) * self.massNum + (j + 1)
            self.gridIndices[grid_id * 12 + 9] = (i + 1) * self.massNum + j
            self.gridIndices[grid_id * 12 + 10] = (i + 1) * self.massNum + (j + 1)
            self.gridIndices[grid_id * 12 + 11] = i * self.massNum + (j + 1)

            self.gridColors[grid_id * 6 + 0] = (1, 0, 0)
            self.gridColors[grid_id * 6 + 1] = (1, 0, 0)
            self.gridColors[grid_id * 6 + 2] = (1, 0, 0)
            self.gridColors[grid_id * 6 + 3] = (1, 0, 0)
            self.gridColors[grid_id * 6 + 4] = (1, 0, 0)
            self.gridColors[grid_id * 6 + 5] = (1, 0, 0)

    # RENDER UPDATE -----------
    @ti.kernel
    def UpdateVertices(self):
        for i, j in ti.ndrange(self.massNum, self.massNum):
            self.vertices[i * self.massNum + j] = self.position[i, j]



@ti.data_oriented
class Collider:
    def __init__(self, radius: ti.f32 = 0.3, center: ti.Vector = [0, 0, 0]):
        # simulation arguments
        self.radius = radius
        self.center = ti.Vector.field(3, dtype=float, shape=(1, ))
        self.center[0] = center


# GLOBAL CONFIG ===============================
# fabric config
fabric0 = Fabric(massNum=8)
fabric1 = Fabric(massNum=24)
fabric2 = Fabric(massNum=64)
fabric3 = Fabric(massNum=256)
sphere = Collider()

# spring config
STIFFNESS = 1e6
DAMPING = 1e2

# environment arguments
GRAVITY = ti.Vector([0, -9.8, 0])
AIR_DRAG = 1.0

# system arguments
dt = 1 / 5000
SUBSTEPS = int(1 / 60 // dt)


# GGUI SETTINGS ================================
window = ti.ui.Window("Fabric Simulation", (1600, 1600))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(0.0, 0.0, 3)
camera.lookat(0.0, 0.0, 0)
scene.set_camera(camera)

fabric = fabric0
currentTime = 0.0
skeletion = False
while window.running:
    # global render
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(sphere.center, 
                    radius=sphere.radius * 0.95, 
                    color=(65/255, 105/255, 225/255))
    
    fabric.UpdateVertices()
    if skeletion:
        scene.particles(fabric.vertices, 
                        color=(106/255, 90/255, 205/255), 
                        radius=0.1 / fabric.massNum)
        scene.lines(fabric.vertices, 
                    indices=fabric.gridIndices, 
                    per_vertex_color=fabric.gridColors, 
                    width=50 / fabric.massNum)
    else:
        scene.mesh(fabric.vertices,
                indices=fabric.triangleIndices,
                per_vertex_color=fabric.triangleColors,
                two_sided=True)

    # keyboard operation
    if window.get_event(ti.ui.PRESS):
        key = window.event.key
        if key == "q":
            exit()
        if key == "c":
            skeletion = not skeletion

        if key == "y":
            fabric = fabric0
            currentTime = 0
            fabric.InitMassPoints()
        if key == "u":
            fabric = fabric1
            currentTime = 0
            fabric.InitMassPoints()
        if key == "i":
            fabric = fabric2
            currentTime = 0
            fabric.InitMassPoints()
        if key == "o":
            fabric = fabric3
            currentTime = 0
            fabric.InitMassPoints()
    
    # reset
    if currentTime > 2.0:
        fabric.InitMassPoints()
        currentTime = 0

    # iteration
    for i in range(SUBSTEPS):
        fabric.UpdateSys()
        currentTime += dt

    canvas.scene(scene)
    window.show()