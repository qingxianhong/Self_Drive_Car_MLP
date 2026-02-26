import math as m
import random as r
from simple_geometry import *
import matplotlib.pyplot as plt
import numpy as np
import time

#activation function for MLP
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 4D MLP
def train_4D_MLP():   
    X = []
    y = []
    path_line_filename = "train4dAll.txt"
    with open(path_line_filename, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = list(map(float, line.split()))
            X.append(tmp[:-1])
            y.append([(tmp[3]+40)/80])
    X = np.array(X)
    y = np.array(y)

    input_size = 3
    hidden_size = 6
    output_size = 1
    learning_rate = 0.01
    epochs = 18000
    np.random.seed(int(time.time()))
    W1 = np.random.rand(input_size, hidden_size)
    b1 = np.random.rand(1, hidden_size)
    W2 = np.random.rand(hidden_size, output_size)
    b2 = np.random.rand(1, output_size)
    for epoch in range(epochs):
        # forward propagation
        hidden_layer_input = np.dot(X, W1) + b1
        hidden_layer_output = sigmoid(hidden_layer_input)
    
        final_input = np.dot(hidden_layer_output, W2) + b2
        final_output = sigmoid(final_input)
    
        # calculate loss (mean squared error)
        loss = np.mean((y - final_output) ** 2)

        # back propagation
        # calculate error
        error = y - final_output
        d_final_output = error * sigmoid_derivative(final_output)

        # calculate hidden layer error
        error_hidden_layer = d_final_output.dot(W2.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # update weights and biases
        W2 += hidden_layer_output.T.dot(d_final_output) * learning_rate
        b2 += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return W1, b1, W2, b2

# 6D MLP
def train_6D_MLP():
    X = []
    y = []
    path_line_filename = "train6dAll.txt"
    with open(path_line_filename, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = list(map(float, line.split()))
            X.append(tmp[:-1])
            y.append([(tmp[5]+40)/80])
    X = np.array(X)
    y = np.array(y)

    input_size = 5
    hidden_size = 7
    output_size = 1
    learning_rate = 0.01
    epochs = 12000
    np.random.seed(int(time.time()))
    W1 = np.random.rand(input_size, hidden_size)
    b1 = np.random.rand(1, hidden_size)
    W2 = np.random.rand(hidden_size, output_size)
    b2 = np.random.rand(1, output_size)
    for epoch in range(epochs):
        # forward propagation
        hidden_layer_input = np.dot(X, W1) + b1
        hidden_layer_output = sigmoid(hidden_layer_input)
    
        final_input = np.dot(hidden_layer_output, W2) + b2
        final_output = sigmoid(final_input)
    
        # calculate loss (mean squared error)
        loss = np.mean((y - final_output) ** 2)

        # back propagation
        # calculate error
        error = y - final_output
        d_final_output = error * sigmoid_derivative(final_output)

        # calculate hidden layer error
        error_hidden_layer = d_final_output.dot(W2.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # update weights and biases
        W2 += hidden_layer_output.T.dot(d_final_output) * learning_rate
        b2 += np.sum(d_final_output, axis=0, keepdims=True) * learning_rate
        W1 += X.T.dot(d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return W1, b1, W2, b2

class Car():
    def __init__(self) -> None:
        self.radius = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def diameter(self):
        return self.radius/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        if self.wheel_min <= angle <= self.wheel_max:
            self.wheel_angle = angle
        elif angle <= self.wheel_min:
            self.wheel_angle = self.wheel_min
        else:
            self.wheel_angle = self.wheel_max
        
    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.radius*1.5))) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground():
    def __init__(self):
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self._readPathLines()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]
        self.W1_4D, self.b1_4D, self.W2_4D, self.b2_4D = train_4D_MLP()
        self.W1_6D, self.b1_6D, self.W2_6D, self.b2_6D = train_6D_MLP()
        self.car = Car()
        self.track_X_4D = []
        self.track_Y_4D = []
        self.state_record_4D = []
        self.write_wheel_4D = []
        self.track_X_6D = []
        self.track_Y_6D = []
        self.state_record_6D = []
        self.write_wheel_6D = []
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    def predictAction_4D(self, state):
        front_dist, right_dist, left_dist = state[0], state[1], state[2]
        X = np.array([[front_dist, right_dist, left_dist]])
        hidden_layer_input = np.dot(X, self.W1_4D) + self.b1_4D
        hidden_layer_output = sigmoid(hidden_layer_input)
        final_input = np.dot(hidden_layer_output, self.W2_4D) + self.b2_4D
        final_output = sigmoid(final_input)
        return (final_output[0][0]*80)

    def predictAction_6D(self, state):
        front_dist, right_dist, left_dist = state[0], state[1], state[2]
        x_pos, y_pos = self.car.getPosition().x, self.car.getPosition().y
        X = np.array([[x_pos, y_pos, front_dist, right_dist, left_dist]])
        hidden_layer_input = np.dot(X, self.W1_6D) + self.b1_6D
        hidden_layer_output = sigmoid(hidden_layer_input)
        final_input = np.dot(hidden_layer_output, self.W2_6D) + self.b2_6D
        final_output = sigmoid(final_input)
        return (final_output[0][0]*80)

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # chack every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state

def run_example_4D():
    p = Playground()
    state = p.reset()
    p.state_record_4D.append(p.state)
    p.write_wheel_4D.append(p.car.wheel_angle)
    while not p.done:
        action = p.predictAction_4D(state)
        state = p.step(action)
        p.track_X_4D.append(p.car.getPosition('center').x)
        p.track_Y_4D.append(p.car.getPosition('center').y)
        p.state_record_4D.append(p.state)
        p.write_wheel_4D.append(p.car.wheel_angle)
    return p.track_X_4D, p.track_Y_4D, p.state_record_4D, p.write_wheel_4D

def run_example_6D():
    p = Playground()
    state = p.reset()
    p.state_record_6D.append(p.state)
    p.write_wheel_6D.append(p.car.wheel_angle)
    while not p.done:
        action = p.predictAction_6D(state)
        state = p.step(action)
        p.track_X_6D.append(p.car.getPosition('center').x)
        p.track_Y_6D.append(p.car.getPosition('center').y)
        p.state_record_6D.append(p.state)
        p.write_wheel_6D.append(p.car.wheel_angle)
    return p.track_X_6D, p.track_Y_6D, p.state_record_6D, p.write_wheel_6D
        
def initial_draw_graph(t):
    # 4D
    plt.subplot(1,2,1)
    x_coordinate = []
    y_coordinate = []
    path_line_filename = "軌道座標點.txt"
    with open(path_line_filename, 'r', encoding='utf-8') as f:
        #first line
        line = f.readline()
        x, y,z= map(float, line.split(','))
        plt.plot(x, y, 'o',color='blue')
        #sec and third lines
        dest_x = []
        dest_y = []
        line = f.readline()
        x1, y1= map(float, line.split(','))
        line = f.readline()
        x2, y2= map(float, line.split(','))
        dest_x.append(x1)
        dest_x.append(x2)
        dest_x.append(x2)
        dest_x.append(x1)
        dest_y.append(y1)
        dest_y.append(y1)
        dest_y.append(y2)
        dest_y.append(y2)
        plt.fill(dest_x, dest_y, 'orange')
        #other lines
        while True:
            line = f.readline()
            if not line:
                break
            x, y= map(float, line.split(','))
            x_coordinate.append(x)
            y_coordinate.append(y)
    plt.plot(x_coordinate, y_coordinate,'green')
    track_X, track_Y, state_record, write_wheel_angle = run_example_4D()
    with open('track_4D.txt', 'w') as f:
        for i in range(len(track_X)):
            f.write(str(state_record[i][0]) + ' ' + str(state_record[i][1]) + ' ' + str(state_record[i][2]) + ' ' + str(write_wheel_angle[i]) + '\n')
    plt.ion()
    plt.show()
    for i in range(len(track_X)):
        dist_str = 'left: '+str(state_record[i][2])+' right: '+str(state_record[i][1])+' front: '+str(state_record[i][0])
        plt.title(dist_str, fontsize=10)
        plt.plot(track_X[i], track_Y[i], 'o',color='red')
        plt.pause(t)
    # 6D
    plt.subplot(1,2,2)
    x_coordinate = []
    y_coordinate = []
    path_line_filename = "軌道座標點.txt"
    with open(path_line_filename, 'r', encoding='utf-8') as f:
        #first line
        line = f.readline()
        x, y,z= map(float, line.split(','))
        plt.plot(x, y, 'o',color='blue')
        #sec and third lines
        dest_x = []
        dest_y = []
        line = f.readline()
        x1, y1= map(float, line.split(','))
        line = f.readline()
        x2, y2= map(float, line.split(','))
        dest_x.append(x1)
        dest_x.append(x2)
        dest_x.append(x2)
        dest_x.append(x1)
        dest_y.append(y1)
        dest_y.append(y1)
        dest_y.append(y2)
        dest_y.append(y2)
        plt.fill(dest_x, dest_y, 'orange')
        #other lines
        while True:
            line = f.readline()
            if not line:
                break
            x, y= map(float, line.split(','))
            x_coordinate.append(x)
            y_coordinate.append(y)
    plt.plot(x_coordinate, y_coordinate,'green')
    track_X, track_Y, state_record, write_wheel_angle = run_example_6D()
    with open('track_6D.txt', 'w') as f:
        for i in range(len(track_X)):
            f.write(str(track_X[i]) + ' ' + str(track_Y[i]) + ' ' + str(state_record[i][0]) + ' ' + str(state_record[i][1]) + ' ' + str(state_record[i][2]) + ' ' + str(write_wheel_angle[i]) + '\n')
    for i in range(len(track_X)):
        dist_str = 'left: '+str(state_record[i][2])+' right: '+str(state_record[i][1])+' front: '+str(state_record[i][0])
        plt.title(dist_str, fontsize=10)
        plt.plot(track_X[i], track_Y[i], 'o',color='red')
        plt.pause(t)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    initial_draw_graph(0.5)