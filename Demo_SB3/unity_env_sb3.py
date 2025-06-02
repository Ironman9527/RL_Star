import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecMonitor
import numpy as np
import zmq
import msgpack
import subprocess
import atexit
import time
import argparse
import os

# --- 观测空间大小和归一化参数 ---
# 1. Agent速度 (XZ)           [2]
# 2. Agent朝向 (Forward XZ)   [2]
# 3. Target归一化方向 (XZ)    [2]
# 4. Target归一化距离         [1]
# 5. Agent朝向与目标方向夹角cos [1]
# 6. Agent朝向与目标方向夹角sin [1] (或直接用角度 / PI)
# 总计 = 9
OBS_FLAT_SIZE = 9 
ACTION_SIZE = 2

MAX_AGENT_SPEED = 10.0 
MAX_TARGET_DISTANCE = 75.0 

def flatten_and_normalize_observation(structured_obs, max_episode_steps_for_env):
    """
    将结构化的观测数据展平并进行归一化，侧重于相对特征和角度。
    假设Unity发送的是世界坐标。
    """
    try:
        agent_data = structured_obs.get('agentData', {})
        
        flat_list = []

        # 1. Agent速度 (XZ平面，世界系，归一化)
        vel_world = agent_data.get('velocity', [0.0]*3)
        flat_list.extend([
            vel_world[0] / MAX_AGENT_SPEED, # X速度
            vel_world[2] / MAX_AGENT_SPEED  # Z速度
        ])

        # 2. Agent朝向 (ForwardVector的XZ分量，已经是单位向量的部分)
        fwd_world = agent_data.get('forwardVector', [0.0, 0.0, 1.0]) # 默认Z轴向前
        agent_fwd_xz = np.array([fwd_world[0], fwd_world[2]], dtype=np.float32)
        norm_agent_fwd_xz = np.linalg.norm(agent_fwd_xz)
        if norm_agent_fwd_xz > 1e-6: # 避免除以零
            agent_fwd_xz /= norm_agent_fwd_xz
        flat_list.extend(agent_fwd_xz.tolist())

        # 3. 到目标的归一化方向向量 (XZ平面，直接使用Unity计算好的)
        # relativeDirectionToTarget已经是归一化的从Agent指向Target的XZ方向
        dir_to_target_xz = np.array(structured_obs.get('relativeDirectionToTarget', [0.0]*2), dtype=np.float32)
        flat_list.extend(dir_to_target_xz.tolist())

        # 4. 到目标的归一化距离
        dist_to_target = structured_obs.get('distanceToTarget', MAX_TARGET_DISTANCE)
        flat_list.append(np.clip(dist_to_target / MAX_TARGET_DISTANCE, 0.0, 1.0))

        # 5. Agent朝向与目标方向的夹角余弦 (cos(angle))
        cos_angle = np.dot(agent_fwd_xz, dir_to_target_xz)
        flat_list.append(np.clip(cos_angle, -1.0, 1.0))

        # 6. Agent朝向与目标方向的夹角正弦 (sin(angle)) - 用于判断左右
        sin_angle = agent_fwd_xz[0] * dir_to_target_xz[1] - agent_fwd_xz[1] * dir_to_target_xz[0]
        flat_list.append(np.clip(sin_angle, -1.0, 1.0))
        
        # 确保最终长度正确
        if len(flat_list) != OBS_FLAT_SIZE:
            print(f"CRITICAL Warning: Flattened observation size mismatch after normalization. Expected {OBS_FLAT_SIZE}, got {len(flat_list)}. Raw Obs: {structured_obs}")
            if len(flat_list) < OBS_FLAT_SIZE: flat_list.extend([0.0] * (OBS_FLAT_SIZE - len(flat_list)))
            else: flat_list = flat_list[:OBS_FLAT_SIZE]
        
        return np.array(flat_list, dtype=np.float32)

    except Exception as e:
        print(f"An unexpected error occurred during F&N: {e}. Observation: {structured_obs}")
        import traceback
        traceback.print_exc() # 打印详细错误堆栈
        return np.zeros(OBS_FLAT_SIZE, dtype=np.float32)


class UnityVectorEnv(VecEnv):
    def __init__(self, executable_path=None, port=5555, num_envs=1,
                 unity_upspeed=1, unity_qps=10, unity_max_steps=1000,
                 launch_unity=True, worker_id=0, editor_debug_mode=False):
        
        self.num_envs = num_envs
        self.port = port + worker_id 
        self.unity_process = None
        self.editor_debug_mode = editor_debug_mode
        self.timeout_seconds = 60 
        self.unity_max_steps_for_normalization = unity_max_steps
        
        single_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_FLAT_SIZE,), dtype=np.float32)
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        
        super().__init__(num_envs=self.num_envs, 
                         observation_space=single_observation_space, 
                         action_space=single_action_space)

        self.current_render_mode = "human" 

        if executable_path and launch_unity and not self.editor_debug_mode:
            unity_exe_path = os.path.abspath(executable_path) 
            if not os.path.exists(unity_exe_path): raise FileNotFoundError(f"Unity executable not found at {unity_exe_path}")
            launch_command = [unity_exe_path, "-ip", "127.0.0.1", "-port", str(self.port),
                              "-numEnvs", str(self.num_envs), "-upSpeed", str(unity_upspeed),
                              "-qps", str(unity_qps), "-maxSteps", str(self.unity_max_steps_for_normalization), 
                              "-testMode", "false", "-logfile", f"unity_env_log_w{worker_id}_p{self.port}.txt",]
            print(f"Python: Launching Unity: {' '.join(launch_command)}")
            try: self.unity_process = subprocess.Popen(launch_command); print(f"Python: Unity PID: {self.unity_process.pid}. Waiting 5s..."); atexit.register(self.close); time.sleep(5) 
            except Exception as e: print(f"Python: Failed to launch Unity: {e}"); raise
        elif self.editor_debug_mode: print(f"Python: Editor Debug Mode. Unity Editor connect to port {self.port}.")
        elif not launch_unity: print(f"Python: Not launching Unity. External instance connect to port {self.port}.")

        self.context = zmq.Context(); self.socket = self.context.socket(zmq.REP) 
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_seconds * 1000) 
        self.socket.setsockopt(zmq.LINGER, 0); print(f"Python: Binding ZMQ REP: tcp://*:{self.port}"); self.socket.bind(f"tcp://*:{self.port}")
        print(f"Python: ZMQ Server listening on port {self.port}..."); self._current_unity_batch_data = None; self._is_first_communication = True


    def _get_target_envs(self, indices):
        """
        辅助函数，用于解析传入的indices参数，返回一个目标环境索引的列表。
        参考自 Stable Baselines3 DummyVecEnv。
        """
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            if not (0 <= indices < self.num_envs):
                raise ValueError(f"Index {indices} out of bounds for {self.num_envs} environments.")
            return [indices]
        if isinstance(indices, (list, tuple, np.ndarray)):
            processed_indices = []
            for index in indices:
                if not (isinstance(index, (int, np.integer)) and 0 <= index < self.num_envs) :
                     raise ValueError(f"Index {index} in indices list is invalid or out of bounds for {self.num_envs} environments.")
                processed_indices.append(int(index))
            return processed_indices
        raise TypeError(f"Unsupported type for indices: {type(indices)}")


    def get_attr(self, attr_name: str, indices = None):
        """
        从矢量化环境中的一个或多个（由indices指定）子环境获取属性。
        """
        target_indices = self._get_target_envs(indices)
        
        if attr_name == "render_mode":
            return [self.current_render_mode for _ in target_indices]
        elif attr_name == "spec":
            return [None for _ in target_indices] 
        else:
            raise NotImplementedError(f"Attribute '{attr_name}' is not handled by UnityVectorEnv.get_attr. "
                                      "You might need to implement it if SB3 requires this attribute "
                                      "from underlying environments.")

    def _receive_unity_batch(self):
        try:
            message_bytes = self.socket.recv()
            if not message_bytes : return False 
            self._current_unity_batch_data = msgpack.unpackb(message_bytes, raw=False)
            if 'envDataList' not in self._current_unity_batch_data or \
               not isinstance(self._current_unity_batch_data['envDataList'], list) or \
               len(self._current_unity_batch_data['envDataList']) != self.num_envs: 
                print(f"Python: Received malformed or incomplete batch data. Expected {self.num_envs}, got {len(self._current_unity_batch_data.get('envDataList', []))}.")
                return False 
            self._current_unity_batch_data['envDataList'].sort(key=lambda x: x.get('envId', -1))
            return True
        except zmq.error.Again: print(f"Python: ZMQ receive timeout after {self.timeout_seconds}s."); return False
        except zmq.error.ContextTerminated: return False
        except Exception as e: print(f"Python: Error receiving/unpacking data: {e}"); return False

    def _send_actions_to_unity(self, actions_list_of_dicts):
        try: self.socket.send(msgpack.packb({'actionsList': actions_list_of_dicts}))
        except zmq.error.ContextTerminated: pass
        except Exception as e: print(f"Python: Error sending actions: {e}")

    def reset(self): 
        if self._is_first_communication:
            if not self._receive_unity_batch():
                print("Python Env (VecEnv reset): Critical error - failed to receive initial batch.")
                return np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
            self._is_first_communication = False
        else:
            noop_actions = [{'envId': self._current_unity_batch_data['envDataList'][i]['envId'] if self._current_unity_batch_data and self._current_unity_batch_data.get('envDataList') and i < len(self._current_unity_batch_data['envDataList']) else i, 
                             'action': [0.0]*ACTION_SIZE, 'nextResetProximityFactor': None} 
                            for i in range(self.num_envs)]
            self._send_actions_to_unity(noop_actions)
            if not self._receive_unity_batch():
                print("Python Env (VecEnv reset): Critical error - failed to receive batch after NOOPs.")
                return np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
        
        obs_list = [flatten_and_normalize_observation(env_data['observation'], self.unity_max_steps_for_normalization) 
                    for env_data in self._current_unity_batch_data['envDataList']]
        return np.array(obs_list, dtype=np.float32)

    def step_async(self, actions: np.ndarray):
        self._actions_to_send_list = []
        for i in range(self.num_envs):
            env_data_ref = self._current_unity_batch_data['envDataList'][i] if self._current_unity_batch_data and self._current_unity_batch_data.get('envDataList') and i < len(self._current_unity_batch_data['envDataList']) else {}
            env_id = env_data_ref.get('envId',i)
            is_done = env_data_ref.get('isDone',False)
            action_for_env_list = actions[i].tolist()
            next_prox_factor = np.random.rand() if is_done and np.random.rand() < 0.5 else None
            self._actions_to_send_list.append({'envId': env_id, 'action': action_for_env_list, 'nextResetProximityFactor': next_prox_factor})
        self._send_actions_to_unity(self._actions_to_send_list)

    def step_wait(self):
        if not self._receive_unity_batch():
            print("Python Env (VecEnv step_wait): Critical error - failed to receive batch.")
            obs = np.zeros((self.num_envs, OBS_FLAT_SIZE), dtype=np.float32)
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.ones(self.num_envs, dtype=bool) 
            infos = [{'error': 'Communication failure'}] * self.num_envs
            return obs, rewards, dones, infos 

        obs_list, reward_list, done_list, infos = [], [], [], [{} for _ in range(self.num_envs)]
        for i, env_data in enumerate(self._current_unity_batch_data['envDataList']):
            obs_list.append(flatten_and_normalize_observation(env_data['observation'], self.unity_max_steps_for_normalization))
            reward_list.append(env_data.get('reward', 0.0))
            done_list.append(env_data.get('isDone', True))
        return (np.array(obs_list, dtype=np.float32),
                np.array(reward_list, dtype=np.float32),
                np.array(done_list, dtype=bool), 
                infos)
    
    def step(self, actions): self.step_async(actions); return self.step_wait()
    def close(self):
        print("Python: Close sequence...");
        if hasattr(self, 'socket') and self.socket: print("Python: Closing ZMQ socket."); self.socket.close(linger=0); self.socket = None 
        if hasattr(self, 'context') and self.context: print("Python: Terminating ZMQ context."); self.context.term(); self.context = None
        if hasattr(self, 'unity_process') and self.unity_process and self.unity_process.poll() is None:
            print(f"Python: Terminating Unity (PID: {self.unity_process.pid})...");
            try: self.unity_process.terminate(); self.unity_process.wait(timeout=10)
            except subprocess.TimeoutExpired: print("Python: Unity terminate timeout, killing."); self.unity_process.kill(); self.unity_process.wait(timeout=5)
            except Exception as e: print(f"Python: Error terminating Unity: {e}")
            if self.unity_process.poll() is None : print("Python: Unity still running after kill attempt.") 
            else: print("Python: Unity process terminated.")
        self.unity_process = None; print("Python: Cleanup complete.")

    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None): raise NotImplementedError
    def seed(self, seed=None): 
        if self.action_space is not None: self.action_space.seed(seed)

def evaluate_model(args, env_to_eval, n_eval_episodes=5):
    """
    加载并评估已训练的模型。
    :param args: 包含 model_path 和其他环境参数的命令行参数。
    :param env_to_eval: 用于评估的 UnityVectorEnv 实例。
    :param n_eval_episodes: 评估的回合数。
    """
    from stable_baselines3 import PPO # 仅在评估/训练时导入

    if not args.model_path or not os.path.exists(args.model_path):
        print(f"Python Evaluator: Error - Model path '{args.model_path}' not provided or does not exist.")
        return

    try:
        # 加载模型时，可以传递env，SB3会检查兼容性
        # 如果VecNormalize被用于训练，加载时也需要加载对应的统计数据
        print(f"Python Evaluator: Loading model from {args.model_path}...")
        model = PPO.load(args.model_path, env=env_to_eval)
        print("Python Evaluator: Model loaded successfully.")
    except Exception as e:
        print(f"Python Evaluator: Error loading model: {e}")
        return

    total_rewards_all_envs = [] # 记录所有环境中所有回合的总奖励
    total_lengths_all_envs = [] # 记录所有环境中所有回合的总长度

    # 对于VecEnv，reset返回的是一个批量的观测数据
    current_obs = env_to_eval.reset() 

    # 为每个并行环境追踪当前回合的奖励和长度
    current_episode_rewards = np.zeros(env_to_eval.num_envs, dtype=np.float32)
    current_episode_lengths = np.zeros(env_to_eval.num_envs, dtype=np.int32)
    
    episodes_completed_count = 0

    print(f"\nPython Evaluator: Starting evaluation for {n_eval_episodes * env_to_eval.num_envs} conceptual episodes "
          f"(or until {n_eval_episodes} episodes complete in env 0 if num_envs > 1)...")

    # 循环直到收集到足够的总回合数，或者达到一个总步数上限以防无限循环
    max_total_eval_steps = n_eval_episodes * args.unity_max_steps * env_to_eval.num_envs * 1.5 # 넉넉한 상한선
    current_total_eval_steps = 0

    while episodes_completed_count < n_eval_episodes * env_to_eval.num_envs and current_total_eval_steps < max_total_eval_steps:
        actions, _states = model.predict(current_obs, deterministic=True)
        next_obs, rewards, dones, infos = env_to_eval.step(actions)

        current_obs = next_obs
        current_episode_rewards += rewards
        current_episode_lengths += 1
        current_total_eval_steps += env_to_eval.num_envs


        for i in range(env_to_eval.num_envs):
            if dones[i]:
                env_id_info = infos[i].get('envId', i) # 尝试从info获取，否则用索引
                print(f"  Env {env_id_info} - Episode Finished. Reward: {current_episode_rewards[i]:.2f}, Length: {current_episode_lengths[i]}")
                total_rewards_all_envs.append(current_episode_rewards[i])
                total_lengths_all_envs.append(current_episode_lengths[i])
                
                # 重置该环境的追踪器
                current_episode_rewards[i] = 0
                current_episode_lengths[i] = 0
                episodes_completed_count +=1
                
                # 注意: VecEnv的reset是在一个环境done之后，下一次step返回的obs就是新episode的开始。
                # model.predict会处理好这个新的obs。我们不需要在这里手动调用env_to_eval.reset()的特定索引。

        if episodes_completed_count >= n_eval_episodes * env_to_eval.num_envs : # 如果收集到足够的总回合数
             break


    if total_rewards_all_envs:
        print("\n--- Evaluation Summary ---")
        print(f"Total episodes evaluated (across all envs): {len(total_rewards_all_envs)}")
        print(f"Mean Reward: {np.mean(total_rewards_all_envs):.2f} +/- {np.std(total_rewards_all_envs):.2f}")
        print(f"Mean Length: {np.mean(total_lengths_all_envs):.2f} +/- {np.std(total_lengths_all_envs):.2f}")
    else:
        print("\nNo complete episodes recorded during evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python test/trainer for Unity ZMQ environment.")
    parser.add_argument("--executable-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--unity-upspeed", type=int, default=1)
    parser.add_argument("--unity-qps", type=int, default=5)
    parser.add_argument("--unity-max-steps", type=int, default=1000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--editor-debug", action="store_true")
    parser.add_argument("--train", action="store_true", help="Flag to run PPO training.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--save-freq", type=int, default=500_000)
    parser.add_argument("--log-dir", type=str, default="./ppo_unity_training_logs/")
    parser.add_argument("--model-name", type=str, default="ppo_unity_finder_rel_obs") # 更新模型名
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--evaluate", action="store_true", help="Flag to run model evaluation.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the saved .zip model file for evaluation.")
    parser.add_argument("--n-eval-episodes", type=int, default=100, help="Number of episodes to run for evaluation (per env instance).")

    args = parser.parse_args()

    if args.editor_debug: 
        launch_unity_process = False
        print("Python: Editor debug mode. Unity Editor should be running.")
    elif args.executable_path: 
        launch_unity_process = True
    else: 
        print("Python: Error - Must provide --executable-path OR use --editor-debug flag.")
        exit(1)

    env = None
    try:
        print("Python: Creating UnityVectorEnv...")
        base_env = UnityVectorEnv( # 创建原始的UnityVectorEnv实例
            executable_path=args.executable_path, port=args.port, num_envs=args.num_envs,
            unity_upspeed=args.unity_upspeed, unity_qps=args.unity_qps, 
            unity_max_steps=args.unity_max_steps, 
            launch_unity=launch_unity_process,
            worker_id=args.worker_id,
            editor_debug_mode=args.editor_debug
        )
        print("Python: UnityVectorEnv created.")

        if args.train:
            from stable_baselines3 import PPO 
            from stable_baselines3.common.callbacks import CheckpointCallback

            print("Python: Wrapping environment with VecMonitor...")
            env = VecMonitor(base_env, filename=os.path.join(args.log_dir, "monitor_log"))
            print("Python: Environment wrapped with VecMonitor.")
            os.makedirs(args.log_dir, exist_ok=True)
            model_save_path = os.path.join(args.log_dir, args.model_name)
            checkpoint_save_path = os.path.join(args.log_dir, "checkpoints/")
            os.makedirs(checkpoint_save_path, exist_ok=True)
            
            print("Python Trainer: Creating PPO model...")
            model = PPO( "MlpPolicy", env, verbose=1, tensorboard_log=args.log_dir,
                        learning_rate=args.learning_rate, n_steps=args.n_steps,
                        batch_size=args.batch_size, n_epochs=args.n_epochs,
                        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range
            )
            
            checkpoint_callback = CheckpointCallback(save_freq=max(args.save_freq // args.num_envs, 1), 
                                                     save_path=checkpoint_save_path, name_prefix=args.model_name)
            print(f"Python Trainer: Starting training for {args.total_timesteps} timesteps...")
            model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=True)
            model.save(model_save_path)
            print(f"Python Trainer: Final model saved to {model_save_path}.zip")

        elif args.evaluate:
            # --- 模型评估逻辑 ---
            if not args.model_path:
                parser.error("--model-path is required when --evaluate is set.")
            evaluate_model(args, base_env, n_eval_episodes=args.n_eval_episodes)
        else:
            print("Python: Random action test mode. Initializing by calling reset()...")
            observations = base_env.reset() 
            if isinstance(observations, tuple): 
                observations, _ = observations

            print(f"Python: Initial observations received for {len(observations)} environments.")
            
            for step_num in range(2000): 
                random_actions = env.action_space.sample() 
                next_observations, rewards, terminateds, infos = env.step(random_actions) 
                
                if (step_num + 1) % 100 == 0:
                    print(f"\n--- Random Test Step: {step_num + 1} ---")
                    for i in range(args.num_envs):
                        if i < len(next_observations):
                             print(f"  Env {i}: Obs (first 5): {next_observations[i][:5]}..., Reward: {rewards[i]:.3f}, Terminated: {terminateds[i]}")
                             if terminateds[i] and 'episode' in infos[i]:
                                 print(f"    Env {i} Episode Info: {infos[i]['episode']}")
            print(f"\nPython: Completed random action test steps.")
    except KeyboardInterrupt: print("\nPython: Process interrupted by user.")
    except Exception as e: print(f"Python: An error occurred: {e}"); import traceback; traceback.print_exc()
    finally:
        if env: print("Python: Cleaning up environment..."); env.close()
        print("Python: Script finished.")