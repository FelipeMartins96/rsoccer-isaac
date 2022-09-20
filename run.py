from isaacgym import gymapi

gym = gymapi.acquire_gym()

# Create sim
def get_sim_params():
    p = gymapi.SimParams()
    p.dt=0.01666666753590107
    p.enable_actor_creation_warning=True
    p.flex=gymapi.FlexParams()
    p.gravity=gymapi.Vec3(0.000000, -9.800000, 0.000000)
    p.num_client_threads=0
    def get_physxparams():
        px = gymapi.PhysXParams()
        px.always_use_articulations=False
        px.bounce_threshold_velocity=0.20000000298023224
        px.contact_collection=gymapi.CC_ALL_SUBSTEPS
        px.contact_offset=0.019999999552965164
        px.default_buffer_size_multiplier=2.0
        px.friction_correlation_distance=0.02500000037252903
        px.friction_offset_threshold=0.03999999910593033
        px.max_depenetration_velocity=100.0
        px.max_gpu_contact_pairs=1048576
        px.num_position_iterations=4
        px.num_subscenes=0
        px.num_threads=0
        px.num_velocity_iterations=1
        px.rest_offset=0.0010000000474974513
        px.solver_type=1
        px.use_gpu=False
        return px
    p.physx= get_physxparams()
    p.stress_visualization=False
    p.stress_visualization_max=100000.0
    p.stress_visualization_min=0.0
    p.substeps=2
    p.up_axis=gymapi.UP_AXIS_Z  #Default gymapi.UP_AXIS_Y
    p.use_gpu_pipeline=False
    return p

sim_params = get_sim_params()
sim = gym.create_sim(compute_device=0, graphics_device=0, type=gymapi.SIM_PHYSX, params=sim_params)

# Create viewer
def get_camera_properties():
    c = gymapi.CameraProperties()
    c.enable_tensors=False
    c.far_plane=2000000.0
    c.height=900
    c.horizontal_fov=90.0
    c.near_plane=0.0010000000474974513
    c.supersampling_horizontal=1
    c.supersampling_vertical=1
    c.use_collision_geometry=False
    c.width=1600
    return c

cam_props = get_camera_properties()
viewer = gym.create_viewer(sim, cam_props)

# Create ground plane
def get_plane_params():
    p = gymapi.PlaneParams()
    p.distance=0.0
    p.dynamic_friction=1.0
    p.normal=gymapi.Vec3(0, 0, 1)   # defaultgymapi.Vec3(0.000000, 1.000000, 0.000000)
    p.restitution=0.0
    p.segmentation_id=0
    p.static_friction=1.0
    return p

plane_params = get_plane_params()
gym.add_ground(sim, plane_params)

# Run loop
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)