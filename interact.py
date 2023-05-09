import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy as np
import openmesh as om
from pyrr import Matrix44

from ArcBall import ArcBallUtil


class QGLControllerWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(QGLControllerWidget, self).__init__(parent)
        self.marked_point = None
        

    def initializeGL(self):
        self.ctx = moderngl.create_context()

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                out vec3 v_vert;
                out vec3 v_norm;
                void main() {
                    v_vert = in_position;
                    v_norm = in_normal;
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec4 Color;
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                out vec4 f_color;
                void main() {
                    float lum = dot(normalize(v_norm),
                                    normalize(v_vert - Light));
                    lum = acos(lum) / 3.14159265;
                    lum = clamp(lum, 0.0, 1.0);
                    lum = lum * lum;
                    lum = smoothstep(0.0, 1.0, lum);
                    lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
                    lum = lum * 0.8 + 0.2;
                    vec3 color = Color.rgb * Color.a;
                    f_color = vec4(color * lum, 1.0);
                }
            '''
        )

        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        self.mesh = None
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        self.center = np.zeros(3)
        self.scale = 1.0

    
    def set_mesh(self, mesh):
        self.mesh = mesh
        self.mesh.update_normals()
        assert(self.mesh.n_vertices() > 0 and self.mesh.n_faces() > 0)
        index_buffer = self.ctx.buffer(
            np.array(self.mesh.face_vertex_indices(), dtype="u4").tobytes())
        vao_content = [
            (self.ctx.buffer(
                np.array(self.mesh.points(), dtype="f4").tobytes()),
                '3f', 'in_position'),
            (self.ctx.buffer(
                np.array(self.mesh.vertex_normals(), dtype="f4").tobytes()),
                '3f', 'in_normal')
        ]
        # self.vbo = self.ctx.buffer(self.mesh.points().astype('f4').tobytes())

        self.vao = self.ctx.vertex_array(
                self.prog, vao_content, index_buffer, 4
            )
        # self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

        self.init_arcball()

    def init_arcball(self):
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        pts = self.mesh.points()
        bbmin = np.min(pts, axis=0)
        bbmax = np.max(pts, axis=0)
        self.center = 0.5*(bbmax+bbmin)
        self.scale = np.linalg.norm(bbmax-self.center)
        self.arc_ball.Transform[:3, :3] /= self.scale
        self.arc_ball.Transform[3, :3] = -self.center/self.scale

    def paintGL(self):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        if self.mesh is None:
            return
        
        self.aspect_ratio = self.width()/max(1.0, self.height())
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio,
                                               0.1, 1000.0)
        lookat = Matrix44.look_at(
            (0.0, 0.0, 2.0),  # eye
            (0.0, 0.0, 0.0),  # target
            (0.0, 1.0, 0.0),  # up
        )
        
        if self.marked_point is not None:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.prog['Color'].value = (1.0, 0.0, 0.0, 1.0)  # red
            self.prog['Mvp'].write(
                (proj * lookat * self.arc_ball.Transform @
                 Matrix44.from_translation(self.marked_point)).astype('f4'))
            self.ctx.point_size = 10
            self.vao.render(mode=moderngl.POINTS, vertices=1)


    
        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 0.8)
        self.arc_ball.Transform[3, :3] = \
            -self.arc_ball.Transform[:3, :3].T@self.center
        self.mvp.write(
            (proj * lookat * self.arc_ball.Transform).astype('f4'))

        self.vao.render()

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)
        self.arc_ball.setBounds(width, height)
        return



    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftDown(event.x(), event.y())
        elif event.buttons() & QtCore.Qt.RightButton:
            self.marked_point = self.get_clicked_point(event)
            

    def mouseReleaseEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftUp()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onDrag(event.x(), event.y())

    def get_clicked_point(self, event):
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        x_ndc = 2 * event.x() / self.width() - 1
        y_ndc = 1 - 2 * event.y() / self.height()
        mvp_inv = np.linalg.inv(proj @ lookat @ self.arc_ball.Transform).astype('f4')
        pos_ndc = np.array([x_ndc, y_ndc, 0.0, 1.0], dtype='f4')
        pos_world = mvp_inv @ pos_ndc
        pos_world /= pos_world[3]
        
        # Get all vertices and faces of the mesh
        vertices = np.array([[v[0], v[1], v[2]] for v in self.mesh.points()])
        faces = np.array([[f[0], f[1], f[2]] for f in self.mesh.face_vertex_indices()])
        
        # Compute the distances from the clicked point to all vertices
        distances = np.sqrt(np.sum((vertices - pos_world[:3])**2, axis=1))
        
        # Find the index of the closest vertex
        closest_vertex_index = np.argmin(distances)
        
        print("all mesh points: ", len(self.mesh.points()))
        # Find the position of the closest vertex
        closest_vertex = vertices[closest_vertex_index]
        print("closest vertex: ", closest_vertex)
        return closest_vertex

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.gl = QGLControllerWidget(self)

        self.setCentralWidget(self.gl)
        self.menu = self.menuBar().addMenu("&File")
        self.menu.addAction('&Open', self.openFile)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.gl.updateGL)
        timer.start()

    def openFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', '', "Mesh files (*.obj *.off *.stl *.ply)")
        mesh = om.read_trimesh(fname[0])
        self.gl.set_mesh(mesh)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow()

    win.show()
    app.exec()
