from lib2to3.pgen2.literals import simple_escapes
from isaacgym import gymapi


def print_vars(cl):
    vrs = [v for v in dir(cl) if '__' not in v]

    for v in vrs:
        print(f'{v}: {getattr(cl, v)}')


# gym.create_sim
## gymapi.SimParams() file:///home/fbm2/isaac/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.SimParams
sim_params = gymapi.SimParams()
print('SimParams ------------------')
print_vars(sim_params)
print('----------------------------')
## gymapi.PhysXParams() file:///home/fbm2/isaac/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.PhysXParams
physx_params = gymapi.PhysXParams()
print('PhysXParams ----------------')
print_vars(physx_params)
print('----------------------------')

# gym.add_ground()
## gymapi.PlaneParams() file:///home/fbm2/isaac/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.PlaneParams
plane_params = gymapi.PlaneParams()
print('PlaneParams ----------------')
print_vars(plane_params)
print('----------------------------')

# gym.load_asset()
## gymapi.AssetOptions() file:///home/fbm2/isaac/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.AssetOptions
asset_options = gymapi.AssetOptions()
print('AssetOptions ---------------')
print_vars(asset_options)
print('----------------------------')

# actor

gym = gymapi.acquire_gym()
sim = gym.create_sim(
    compute_device=0, graphics_device=0, type=gymapi.SIM_PHYSX, params=sim_params
)
gym.add_ground(sim, plane_params)
spacing = 1.0
lower = gymapi.Vec3(-spacing, -spacing, 0)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

rbt_asset = gym.load_asset(
    sim=sim, rootpath='./assets/', filename='vss_robot.urdf', options=asset_options
)
rbt_pose = gymapi.Transform(p=gymapi.Vec3(0, 0.1, 0.024))
actor = gym.create_actor(
    env=env, asset=rbt_asset, pose=rbt_pose, group=0, filter=0b00, name='robot'
)

## gym.get_actor_rigid_shape_properties() file:///home/fbm2/isaac/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.RigidShapeProperties
rigid_shape_properties = gym.get_actor_rigid_shape_properties(env, actor)
for i, prop in enumerate(rigid_shape_properties):
    print(f'rigidshapeprops {i} ----------')
    print_vars(rigid_shape_properties[i])
    print('----------------------------')
## gym.get_actor_rigid_body_properties() file:///home/fbm2/isaac/isaacgym/docs/api/python/struct_py.html#isaacgym.gymapi.RigidBodyProperties
rigid_body_properties = gym.get_actor_rigid_body_properties(env, actor)
for i, prop in enumerate(rigid_body_properties):
    print(f'rigidbodyprops {i} -----------')
    print_vars(rigid_body_properties[i])
    print('----------------------------')
## gym.get_actor_dof_properties() file:///home/fbm2/isaac/isaacgym/docs/api/python/gym_py.html?highlight=dof_properties#isaacgym.gymapi.Gym.get_actor_dof_properties
dof_properties = gym.get_actor_dof_properties(env, actor)
print(f'dof properties -----------')
for props in dof_properties.dtype.names:
    print(f'{props}: {dof_properties[0][props]}')
print('----------------------------')
