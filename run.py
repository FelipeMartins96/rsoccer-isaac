from isaacgym import gymapi

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim = gym.create_sim(params=sim_params, type=gymapi.SIM_PHYSX)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!

# create the ground plane
gym.add_ground(sim, plane_params)

while True:
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)