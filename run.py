from isaacgym import gymapi

gym = gymapi.acquire_gym()

# Create sim
def get_sim_params():
    p = gymapi.SimParams()
    p.dt=0.01666666753590107
    p.enable_actor_creation_warning=True
    p.flex=gymapi.FlexParams()
    p.gravity=gymapi.Vec3(0.0, 0.0, -9.8) # default gymapi.Vec3(0.000000, -9.800000, 0.000000)
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
        px.use_gpu=True
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

cam_pos = gymapi.Vec3(0.0, -0.2, 4)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

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

# Create env
spacing = 1.0
lower = gymapi.Vec3(-spacing, -spacing, 0)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

# Add actor
def get_asset_options():
    a = gymapi.AssetOptions()
    a.angular_damping=0.5
    a.armature=0.0
    a.collapse_fixed_joints=False
    a.convex_decomposition_from_submeshes=False
    a.default_dof_drive_mode=0
    a.density=1000.0
    a.disable_gravity=False
    a.enable_gyroscopic_forces=True
    a.fix_base_link=False
    a.flip_visual_attachments=False
    a.linear_damping=0.0
    a.max_angular_velocity=64.0
    a.max_linear_velocity=1000.0
    a.mesh_normal_mode=gymapi.FROM_ASSET
    a.min_particle_mass=9.999999960041972e-13
    a.override_com=False
    a.override_inertia=False
    a.replace_cylinder_with_capsule=False
    a.slices_per_cylinder=100 #default 20
    a.tendon_limit_stiffness=1.0
    a.thickness=0.019999999552965164
    a.use_mesh_materials=False
    a.use_physx_armature=True
    a.vhacd_enabled=False
    a.vhacd_params=gymapi.VhacdParams()
    return a

def add_robot(group=0):
    options = gymapi.AssetOptions()
    rbt_asset = gym.load_asset(sim=sim, rootpath='./assets/', filename='vss_robot.urdf', options=options)
    rbt_initial_height = 0.028 # Z dimension
    rbt_pose = gymapi.Transform(p=gymapi.Vec3(0, 0.1, rbt_initial_height))
    actor = gym.create_actor(env=env, asset=rbt_asset,pose=rbt_pose, group=group, filter=0b0, name='robot')

    props = gym.get_actor_rigid_shape_properties(env, actor)
    body, left_wheel, right_wheel = 0, 1, 2
    props[body].friction = 0.0
    props[body].filter = 0b0
    props[left_wheel].filter = 0b1
    props[right_wheel].filter = 0b1
    gym.set_actor_rigid_shape_properties(env, actor, props)

    props = gym.get_actor_dof_properties(env, actor)
    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
    props["stiffness"].fill(0.0)
    props["damping"].fill(200.0)
    gym.set_actor_dof_properties(env, actor, props)

def add_field(group=0, filter=0b1):
    # Using procedural assets because with an urdf file rigid contacts were not being drawn
    # Height (x), Width (Y), Depth (Z)
    totalWidth = 2.0    # personal choice
    totalHeight = 1.5   # personal choice
    fieldWidth = 1.5    
    fieldHeight = 1.3    
    goalWidth = 0.1     
    goalHeight = 0.4    
    wallsDepth = 0.1    # on rules its 0.05

    options = get_asset_options()
    options.fix_base_link = True
    color = gymapi.Vec3(0.2, 0.2, 0.2)

    # Side Walls (sw)
    def add_side_walls():
        swWidth = totalWidth
        swHeight = (totalHeight - fieldHeight) / 2      
        swX = 0
        swY = (fieldHeight + swHeight) / 2 
        swZ = wallsDepth/2
        swDirs = [(1, 1), (1, -1)] # Top and Bottom

        swAsset = gym.create_box(sim, swWidth, swHeight, wallsDepth, options)

        for dirX, dirY in swDirs:
            swPose = gymapi.Transform(p=gymapi.Vec3(dirX * swX, dirY * swY, swZ))
            swActor = gym.create_actor(env=env, asset=swAsset,pose=swPose, group=group, filter=filter)
            gym.set_rigid_body_color(env, swActor, 0, gymapi.MESH_VISUAL, color)
    
    # End Walls (ew)
    def add_end_walls():
        ewWidth = (totalWidth - fieldWidth) / 2
        ewHeight = (fieldHeight - goalHeight) / 2
        ewX = (fieldWidth + ewWidth) / 2
        ewY = (fieldHeight - ewHeight) / 2
        ewZ = wallsDepth/2
        ewDirs = [(-1, 1), (1, 1), (-1, -1), (1, -1)] # Corners

        ewAsset = gym.create_box(sim, ewWidth, ewHeight, wallsDepth, options)

        for dirX, dirY in ewDirs:
            ewPose = gymapi.Transform(p=gymapi.Vec3(dirX * ewX, dirY * ewY, ewZ))
            ewActor = gym.create_actor(env=env, asset=ewAsset,pose=ewPose, group=group, filter=filter)
            gym.set_rigid_body_color(env, ewActor, 0, gymapi.MESH_VISUAL, color)
    
    # Goal Walls (gw)
    def add_goal_walls():
        gwWidth = ((totalWidth - fieldWidth) / 2) - goalWidth
        gwHeight = goalHeight
        gwX = (totalWidth - gwWidth) / 2
        gwY = 0
        gwZ = wallsDepth/2
        gwDirs = [(-1, 1), (1, 1)] # left and right

        gwAsset = gym.create_box(sim, gwWidth, gwHeight, wallsDepth, options)

        for dirX, dirY in gwDirs:
            gwPose = gymapi.Transform(p=gymapi.Vec3(dirX * gwX, dirY * gwY, gwZ))
            gwActor = gym.create_actor(env=env, asset=gwAsset,pose=gwPose, group=group, filter=filter)
            gym.set_rigid_body_color(env, gwActor, 0, gymapi.MESH_VISUAL, color)

    add_side_walls()
    add_end_walls()
    add_goal_walls()

add_robot()
add_field()

robot_handle = gym.find_actor_handle(env, 'robot')
lwh = gym.find_actor_dof_handle(env, robot_handle, 'body_leftWheel')
rwh = gym.find_actor_dof_handle(env, robot_handle, 'body_rightWheel')


i = 0
# Run loop
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    if i > 60:
        gym.set_dof_target_velocity(env, lwh, 17.0)
        gym.set_dof_target_velocity(env, rwh, 17.0)
    i += 1

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)
    gym.draw_env_rigid_contacts(viewer, env, gymapi.Vec3(1.0,0.0,0.0), 10, True) 

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)