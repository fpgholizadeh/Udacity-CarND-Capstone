from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base,
                 steer_ratio, max_lat_accel, max_steer_angle):
        rospy.logdebug("[Controller] Twist controller ........")
        # Yaw controller
        self.yaw_controller = YawController(
            wheel_base, steer_ratio, .1, max_lat_accel, max_steer_angle)

        # Throttle controller
        kp = .3
        ki = .1
        kd = .0
        min_throttle = .0
        max_throttle = .2
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)

        # LPF for velocity
        tau = .5  # cutoff time constant
        ts = .02  # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        # init param
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.last_time = rospy.get_time()

    def control(self, current_vel, angular_vel, linear_vel, dbw_enabled):
        # Check the dbw status
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        # smthing o current velocity
        current_vel = self.vel_lpf.filt(current_vel)

        # steering control
        steer = self.yaw_controller.get_steering(
            linear_vel, angular_vel, current_vel)

        # throttle control
        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)

        # brake control
        if linear_vel < .1 and vel_error < 0:
            throttle = 0
            brake = 700  # Nm needed to hold the car in plance when we are stopped to a light. Value tuned for Carla simulator
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        else:
            brake = 0

        print("throttle, brake, steer: ", throttle, brake, steer)
        return throttle, brake, steer
