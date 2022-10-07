import argparse
import os


def write_front(args, count):
    print(f"The system is {args.sys}")
    front = f"""#!/bin/bash

"""
    return front


if __name__ == "__main__":

    envs_list = [
        # (env_name, fixed_rollout_len, noise_dim, alpha)
        ('halfcheetah-medium-expert-v2', 5, 50, 0.5),
        ('hopper-medium-expert-v2', 3, 50, 0.5),
        ('walker2d-medium-expert-v2', 5, 50, 0.5),
        ('halfcheetah-medium-v2', 1, 50, 0.5),
        ('hopper-medium-v2', 1, 50, 0.5),
        ('walker2d-medium-v2', 3, 50, 0.5),
        ('halfcheetah-medium-replay-v2', 3, 50, 0.5),
        ('hopper-medium-replay-v2', 1, 50, 0.5),
        ('walker2d-medium-replay-v2', 5, 50, 0.5),
        ('maze2d-umaze-v1', 3, 0, 0.2),
        ('maze2d-medium-v1', 3, 50, 0.2),
        ('maze2d-large-v1', 1, 50, 0.2),
        ('pen-cloned-v1', 5, 1, 0.2),
        ('pen-expert-v1', 1, 50, 0.2),
        ('door-expert-v1', 1, 0, 0.2),
        ('pen-human-v1', 1, 1, 0.2)
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", type=str, default='server')                                            # ['server', 'frontera', 'maverick2']
    parser.add_argument("--num_seed", type=int, default=5)                                              #
    parser.add_argument("--seed", type=int, default=0)  #
    parser.add_argument("--submit_job", type=str, default="False")                                      #
    parser.add_argument("--num_thread", default=4, type=int)                                            #
    parser.add_argument("--device", type=str, default="0,1,2,3")  #

    parser.add_argument("--ExpID", default=1, type=int)                                                 # Experiment ID
    parser.add_argument("--total_iters", default=1e6, type=float)                                       # total number of iterations for MB-SGD training

    parser.add_argument('--actor_lr', default=2e-4, type=float)                                         # actor learning rate
    parser.add_argument('--critic_lr', default=3e-4, type=float)                                        # critic learning rate
    parser.add_argument('--dis_lr', default=2e-4, type=float)                                           # act navigator learning rate
    parser.add_argument('--model_lr', default=0.001, type=float)                                        # act navigator learning rate (was: 3e-4)
    parser.add_argument('--batch_size', default=512, type=int)                                          # batch size of MB-SGD
    parser.add_argument('--log_lagrange', default=10.0, type=float)                                      # maximum value of log lagrange multiplier
    parser.add_argument('--log_lagrange_am', default=10.0, type=float)                                   # 10.0 maximum value of log lagrange multiplier
    parser.add_argument('--log_lagrange_human', default=10.0, type=float)                                # 15.0 maximum value of log lagrange multiplier
    parser.add_argument('--policy_freq', default=2, type=int)                                           # update frequency of the actor

    parser.add_argument('--noise_type', default='normal', type=str)                                     # noise type
    parser.add_argument('--noise_method', default="concat", type=str)                                   # method to add noise in the implicit policy
    parser.add_argument("--noise_dim", default=0, type=int)                                             # dimension of noise in the implicit policy
    parser.add_argument('--sigma', default=1.0, type=float)                                             # standard deviation of the normal noise
    parser.add_argument('--sigma_human', default=1e-3, type=float)                                      # standard deviation of the normal noise
    parser.add_argument('--lower', default=0.0, type=float)                                             # lower bound of uniform noise
    parser.add_argument('--upper', default=1.0, type=float)                                             # upper bound of uniform noise
    parser.add_argument('--warm_start_epochs', default=40, type=int)                                    # number of epochs for warm start training
    parser.add_argument('--state_noise_std', default=-1., type=float)
    parser.add_argument('--num_action_bellman', default=1, type=int)
    parser.add_argument('--rollout_generation_freq', default=250, type=int)
    parser.add_argument('--rollout_batch_size', default=2048, type=int)
    parser.add_argument('--num_model_rollouts', default=128, type=int)
    parser.add_argument('--rollout_retain_epochs', default=5, type=int)
    parser.add_argument("--model_spectral_norm", default='False', type=str)                             # Whether apply spectral norm to every hidden layer (default: False)
    parser.add_argument('--fixed_rollout_len', default=1, type=int)
    parser.add_argument('--real_data_pct', default=0.5, type=float)
    parser.add_argument('--model_epochs_since_last_update', default=10, type=int)
    parser.add_argument('--replay_buffer_device', default="cuda", type=str)  # {"cuda", "numpy"}
    parser.add_argument('--model_retrain_period', default=100, type=int)                                # number of epochs before retraining the model using MIS weights
    parser.add_argument('--density_ratio_grad_steps', default=10e4, type=float)                         # number of gradient steps to train the dr model
    parser.add_argument("--weight_output_clipping", default='False', type=str)                           # Whether perform output_clipping in the weight network
    parser.add_argument("--weight_training_penalty", default='True', type=str)                         #
    parser.add_argument("--weight_training_loss", default='Huber', type=str)                               # {"l1", "l2"}
    parser.add_argument("--weight_training_decay", default=-1., type=float)                               # weight_decay in training the density ratio
    parser.add_argument("--smoothing_power_alpha", default=0.2, type=float)                                             # smoothing power
    parser.add_argument("--q_dis_model", default='False', type=str)
    parser.add_argument("--dr_method", default='TD-DICE', type=str)                                    # {'TD-DICE' (default), "VPM", "GenDICE", "DualDICE"}
    parser.add_argument("--weighted_policy_training", default='False', type=str)                        # Whether use weighted policy training
    parser.add_argument("--use_kl_dual", default='False', type=str)  # Whether use kl dual instead of JSD
    parser.add_argument("--use_weight_wpr", default='True', type=str)  # Whether use weight in weighted policy training (for KL dual)
    parser.add_argument("--use_gaussian_policy", default='False', type=str)  # Whether use Gaussian policy instead of the implicit policy
    parser.add_argument("--remove_reg", default='False', type=str)          # remove regularization
    parser.add_argument("--use_reward_test_func", default='False', type=str)  # use reward as the test function in training the MIW
    args = parser.parse_args()

    if not os.path.exists("./job_scripts"):
        os.makedirs("./job_scripts")
    if not os.path.exists(f"./python_outputs/Exp{args.ExpID}"):
        os.makedirs(f"./python_outputs/Exp{args.ExpID}")

    job_list = []
    gpu_devive = args.device.split(",")
    print(f"Device: {gpu_devive}")
    for idx, (env_name, fixed_rollout_len, noise_dim, alpha) in enumerate(envs_list):
        log_lagrange = args.log_lagrange
        sigma = args.sigma

        if (('antmaze' in env_name) or ('expert' in env_name)) and ("medium" not in env_name):
            log_lagrange = args.log_lagrange_am
            print(f"{env_name}: log_lagrange={log_lagrange}")
        if ('human' in env_name) or ('cloned' in env_name):
            sigma = args.sigma_human
            log_lagrange = args.log_lagrange_human
            print(f"{env_name}: sigma={sigma}, log_lagrange={log_lagrange}")

        if idx % len(gpu_devive) == 0:
            job_file_name = f"./job_scripts/run_Exp{args.ExpID}_{idx // len(gpu_devive) + 1}.sh"
            job_list.append(job_file_name)
            f = open(job_file_name, "w")
            f.write(write_front(args=args, count=(idx // len(gpu_devive) + 1)))
            run_cmd = f"for ((i={args.seed};i<{args.seed + args.num_seed};i+=1)) \n"
            run_cmd += "do \n"

        run_cmd += f"  OMP_NUM_THREADS={args.num_thread} MKL_NUM_THREADS={args.num_thread} python gan_main.py \\\n"
        run_cmd += f"  --device 'cuda:{gpu_devive[idx % len(gpu_devive)]}' \\\n"
        run_cmd += f'  --log_dir "./results/" \\\n'
        run_cmd += f"  --batch_size {args.batch_size} \\\n"
        run_cmd += f'  --env_name "{env_name}" \\\n'
        run_cmd += f"  --policy_freq {args.policy_freq} \\\n"
        run_cmd += f"  --sigma {sigma} \\\n"
        run_cmd += '  --seed $i \\\n'
        run_cmd += f"  --total_iters {args.total_iters} \\\n"
        run_cmd += f"  --dis_lr {args.dis_lr} \\\n"
        run_cmd += f"  --actor_lr {args.actor_lr} \\\n"
        run_cmd += f"  --critic_lr {args.critic_lr} \\\n"
        run_cmd += f"  --model_lr {args.model_lr} \\\n"
        run_cmd += f"  --log_lagrange {log_lagrange} \\\n"
        run_cmd += f"  --noise_dim {noise_dim} \\\n"
        run_cmd += f"  --noise_type {args.noise_type} \\\n"
        run_cmd += f"  --noise_method {args.noise_method} \\\n"
        run_cmd += f"  --lower {args.lower} \\\n"
        run_cmd += f"  --upper {args.upper} \\\n"
        run_cmd += f"  --warm_start_epochs {args.warm_start_epochs} \\\n"
        run_cmd += f"  --state_noise_std {args.state_noise_std} \\\n"
        run_cmd += f"  --num_action_bellman {args.num_action_bellman} \\\n"
        run_cmd += f"  --rollout_generation_freq {args.rollout_generation_freq} \\\n"
        run_cmd += f"  --rollout_batch_size {args.rollout_batch_size} \\\n"
        run_cmd += f"  --num_model_rollouts {args.num_model_rollouts} \\\n"
        run_cmd += f"  --rollout_retain_epochs {args.rollout_retain_epochs} \\\n"
        run_cmd += f"  --model_spectral_norm {args.model_spectral_norm} \\\n"
        run_cmd += f"  --fixed_rollout_len {fixed_rollout_len} \\\n"
        run_cmd += f"  --real_data_pct {args.real_data_pct} \\\n"
        run_cmd += f"  --model_epochs_since_last_update {args.model_epochs_since_last_update} \\\n"
        run_cmd += f"  --replay_buffer_device {args.replay_buffer_device} \\\n"
        run_cmd += f"  --model_retrain_period {args.model_retrain_period} \\\n"
        run_cmd += f"  --density_ratio_grad_steps {args.density_ratio_grad_steps} \\\n"
        run_cmd += f"  --weight_output_clipping {args.weight_output_clipping} \\\n"
        run_cmd += f"  --weight_training_penalty {args.weight_training_penalty} \\\n"
        run_cmd += f"  --weight_training_loss {args.weight_training_loss} \\\n"
        run_cmd += f"  --weight_training_decay={args.weight_training_decay} \\\n"
        run_cmd += f"  --smoothing_power_alpha={alpha} \\\n"
        run_cmd += f"  --q_dis_model={args.q_dis_model} \\\n"
        run_cmd += f"  --dr_method={args.dr_method} \\\n"
        run_cmd += f"  --weighted_policy_training={args.weighted_policy_training} \\\n"
        run_cmd += f"  --use_kl_dual={args.use_kl_dual} \\\n"
        run_cmd += f"  --use_weight_wpr={args.use_weight_wpr} \\\n"
        run_cmd += f"  --use_gaussian_policy={args.use_gaussian_policy} \\\n"
        run_cmd += f"  --remove_reg={args.remove_reg} \\\n"
        run_cmd += f"  --use_reward_test_func={args.use_reward_test_func} \\\n"
        run_cmd += f"  --ExpID {args.ExpID}"
        run_cmd += f' > "./python_outputs/Exp{args.ExpID}/task{idx // len(gpu_devive) + 1}{idx % len(gpu_devive) + 1}_$i.txt" & \n'
        run_cmd += "  sleep 10s \n"

        if idx % len(gpu_devive) == (len(gpu_devive) - 1):
            run_cmd += "  sleep 60s \n"
            run_cmd += "done \n"
            run_cmd += "wait \n"
            run_cmd += "\n"
            run_cmd += "printf '\\nEnd of running...\\n'\n"
            run_cmd += "#########################################################################"

            f.write(run_cmd)
            f.close()

    if args.submit_job == "True":
        for job in job_list:
            print(f"submit job: {job}")
            os.system(f"bash {job}")
    else:
        print(*job_list, sep="\n")
