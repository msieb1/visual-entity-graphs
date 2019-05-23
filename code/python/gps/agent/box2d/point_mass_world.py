""" This file defines an environment for the Box2D PointMass simulator. """
import numpy as np
import Box2D as b2
from framework import Framework

from gps.agent.box2d.settings import fwSettings
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES

class PointMassWorld(Framework):
    """ This class defines the point mass and its environment."""
    name = "PointMass"
    def __init__(self, x0, target, render):
        self.render = render
        if self.render:
            super(PointMassWorld, self).__init__()
        else:
            self.world = b2.b2World(gravity=(0, -10), doSleep=True)
        self.world.gravity = (0.0, 0.0)
        self.initial_position = (x0[0], x0[1])
        self.initial_angle = b2.b2_pi
        self.initial_linear_velocity = (x0[2], x0[3])
        self.initial_angular_velocity = 0

        ground = self.world.CreateBody(position=(0, 20))
        ground.CreateEdgeChain(
            [(-20, -20),
             (-20, 20),
             (20, 20),
             (20, -20),
             (-20, -20)]
            )

        xf1 = b2.b2Transform()
        xf1.angle = 0.3524 * b2.b2_pi
        xf1.position = b2.b2Mul(xf1.R, (1.0, 0.0))

        xf2 = b2.b2Transform()
        xf2.angle = -0.3524 * b2.b2_pi
        xf2.position = b2.b2Mul(xf2.R, (-1.0, 0.0))
        self.body = self.world.CreateDynamicBody(
            position=self.initial_position,
            angle=self.initial_angle,
            linearVelocity=self.initial_linear_velocity,
            angularVelocity=self.initial_angular_velocity,
            angularDamping=5,
            linearDamping=0.1,
            shapes=[b2.b2PolygonShape(vertices=[xf1*(-1, 0),
                                                xf1*(1, 0), xf1*(0, .5)]),
                    b2.b2PolygonShape(vertices=[xf2*(-1, 0),
                                                xf2*(1, 0), xf2*(0, .5)])],
            shapeFixture=b2.b2FixtureDef(density=1.0),
        )
        self.target = self.world.CreateStaticBody(
            position=target[:2],
            angle=self.initial_angle,
            shapes=[b2.b2PolygonShape(vertices=[xf1*(-1, 0), xf1*(1, 0),
                                                xf1*(0, .5)]),
                    b2.b2PolygonShape(vertices=[xf2*(-1, 0), xf2*(1, 0),
                                                xf2*(0, .5)])],
        )
        self.target.active = False

    def run(self):
        """Initiates the first time step
        """
        if self.render:
            super(PointMassWorld, self).run()
        else:
            self.run_next(None)

    def run_next(self, action):
        """Moves forward in time one step. Calls the renderer if applicable."""
        if self.render:
            super(PointMassWorld, self).run_next(action)
        else:
            if action is not None:
                self.body.linearVelocity = (action[0], action[1])
            self.world.Step(1.0 / fwSettings.hz, fwSettings.velocityIterations,
                            fwSettings.positionIterations)

    def Step(self, settings, action):
        """Called upon every step. """
        self.body.linearVelocity = (action[0], action[1])

        super(PointMassWorld, self).Step(settings)

    def reset_world(self):
        """ This resets the world to its initial state"""
        self.world.ClearForces()
        self.body.position = self.initial_position
        self.body.angle = self.initial_angle
        self.body.angularVelocity = self.initial_angular_velocity
        self.body.linearVelocity = self.initial_linear_velocity

    def get_state(self):
        """ This retrieves the state of the point mass"""
        state = {END_EFFECTOR_POINTS: np.append(np.array(self.body.position), [0]),
                 END_EFFECTOR_POINT_VELOCITIES: np.append(np.array(self.body.linearVelocity), [0])}

        return state
