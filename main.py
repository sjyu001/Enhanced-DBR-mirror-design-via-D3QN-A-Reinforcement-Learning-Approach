# main.py
from env import CustomEnv
from network import Qnet, init_params, merge_network_weights, train_network
from replaybuffer import ReplayBuffer
import logger
#from logger import get_logger
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import network
import logging
import time
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
        
    # parameter
    epilen = 30
    stepnum = 5E+6
    minimum_epsilon = 0.001
    eps_greedy_period = 2E+6
    train_start_memory_size = 10000
    train_step = 2
    train_num = 1
    batch_size = 512
    gamma = 0.99
    merge_step = 20000
    tau = 0.2
    printint = 50
    val_num = 10
    size = 15

    t = datetime.datetime.now()
    timeFolderName = t.strftime("%Y_%m_%d_%H_%M_%S")
    
    filepath = 'experiments/' + '/'+timeFolderName

    print('\n File location folder is: %s \n' %filepath)

    os.makedirs(filepath+'/devices', exist_ok=True)
    os.makedirs(filepath+'/np_struct', exist_ok=True)
    os.makedirs(filepath+'/model', exist_ok=True)
    os.makedirs(filepath+'/summary', exist_ok=True)
    os.makedirs(filepath+'/logs', exist_ok=True)


    env = CustomEnv(size=size)
    env_val = CustomEnv(size=size)

    memory = ReplayBuffer(buffer_limit=10000)
    q = network.Qnet(ncells=size)
    q_target = network.Qnet(ncells=size)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=0.01)
    double = True

    epi_len_st= []
    n_epi =1    # start from 1st episode   
    count = 0
    result_flag = 0
    
    path_logs = '/logs/'
    path_np_struct_max = '/np_struct/'
    path_devices_max = '/devices/'
    path_model = '/model/'
    path_summary = '/summary/'
    path_devices = '/devices/epi{}.png'
    path_np_struct = '/np_struct/epi{}.npy'
    path_tensorboard = '/tensorboard/'

    #logger handler
    loggername = filepath+path_logs+'log'
    lgr = logging.getLogger(loggername)
    sh = logging.StreamHandler()
    lgr.addHandler(sh)
    lgr.setLevel(logging.DEBUG)

    ### tensorboard
    writer = SummaryWriter(filepath+path_tensorboard)


    #initialize the saved np arrays    
    x_step = np.array([])
    x_episode = np.array([])    
    one_step_average_reward = np.array([])
    final_step_efficiency = np.array([])
    episode_length_ = np.array([])
    epsilon_ = np.array([])
    max_efficiency = np.array([])
    memory_size = np.array([])
    train_loss = np.array([])
    result_val_mean_np = np.array([])
    result_val_max_np = np.array([])
    result_val_std_np = np.array([])
    result_val_zero_np = np.array([])
    result_val_max_zero_np = np.array([])
    epi_len_val_zero_np = np.array([])

    init_time = time.process_time()
    save_optimum = True

    #Overall Training Process
    while(True):
        s, result_init = env.reset()
        done = False
        result_epi_st = np.zeros((epilen, 1))
        epi_length = 0
        average_reward = 0.0
        
        if count > stepnum:
            break
        
        for t in range(epilen):
            

            # when training, make the minimum epsilon as 1%(can vary by the minimum_epsilon value) for exploration 
            epsilon = max(minimum_epsilon, 0.9 * (1. - count / eps_greedy_period))
            q.eval()
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, result_next, r, done = env.step(a)
            done_mask = 1 - done
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime
            result_epi_st[t] = result_next
            average_reward += r
            epi_length = t+1
            count += 1

            tb = True
            if tb==True:
                #if epi_length!=0:
                writer.add_scalar('final step result / episode', result_next, n_epi)
                writer.add_scalar('max result / episode', result_flag, count)

                if (memory.size() > int(train_start_memory_size)
                and count % int(train_step) == 0):
                    writer.add_scalar('train loss / step', loss, count)
        
            if save_optimum == True: 
                if result_next>result_flag:
                    result_flag = result_next
                    plt.figure(figsize=(20,10))
                    plt.imshow(s.reshape(1,-1), cmap = 'Greys')
                    plt.yticks([])
                    plt.savefig(filepath+path_devices_max+'max.png', format = 'png')
                    plt.savefig(filepath+path_devices_max+'max.eps', format = 'eps')
                    np.save(filepath+path_np_struct_max+'max_structure.npy', s)
                    np.save(filepath+path_np_struct_max+'max_efficiency.npy',np.array(result_flag))
                    np.save(filepath+path_np_struct_max+'max_stepnumber.npy', np.array(count))
                    plt.close()

            if (memory.size() > train_start_memory_size
                and count % train_step == 0):
                
                q.train()
                loss = network.train_network(q, q_target, memory, optimizer, train_num, \
                    batch_size, gamma, double=double)

                if count % merge_step == 0:

                    network.merge_network_weights(q_target.named_parameters(),
                                        q.named_parameters(), tau)
                
            if done:
                break

        if n_epi % printint == 0 and n_epi != 0:

            epsilon_val = 0.01
            max_result_val = 0
            result_epi_st_val = np.zeros((val_num, 1))
            max_result_st_val = np.zeros((val_num, 1))
            _, _ = env_val.reset()
            
            #run episode 10 times
            for i in range(val_num):
                s, _ = env_val.reset()
                for t in range(epilen):
                    q.eval()
                    a = q.sample_action(torch.from_numpy(s).float(), epsilon_val)
                    s_prime, result_next_val, r, done = env_val.step(a)
                    if result_next_val>max_result_val:
                        max_result_val = result_next_val
                    s = s_prime
                max_result_st_val[i] = max_result_val
            
            result_val_mean = np.mean(max_result_st_val)
            result_val_max = np.max(max_result_st_val)
            result_val_std = np.std(max_result_st_val)
            
            # epsilon zero
            epsilon_val_zero = 0
            max_result_val_zero = 0
            epi_len_val_zero = 0

            s, _ = env_val.reset()
            for t in range(epilen):
                q.eval()
                a = q.sample_action(torch.from_numpy(s).float(), epsilon_val_zero)
                s_prime, result_next_val_zero, r, done = env_val.step(a)
                if result_next_val_zero>max_result_val_zero:
                    max_result_val_zero = result_next_val_zero
                s = s_prime
                epi_len_val_zero +=1
                
                if done:
                    break
                
            result_val_zero = result_next_val_zero
            result_val_max_zero = max_result_val_zero
    
            x_step = np.append(x_step, count)
            x_episode = np.append(x_episode, n_epi)
            if epi_length!=0:
                one_step_average_reward = np.append(one_step_average_reward, average_reward/epi_length)
            final_step_efficiency = np.append(final_step_efficiency, result_next)
            episode_length_ = np.append(episode_length_, epi_length)
            epsilon_ = np.append(epsilon_, epsilon*100)
            max_efficiency = np.append(max_efficiency, result_flag)
            memory_size = np.append(memory_size, memory.size())
            if (memory.size() > train_start_memory_size
                and count % train_step == 0):
                loss_numpy = loss.detach().numpy()
                train_loss = np.append(train_loss, loss_numpy)
            result_val_mean_np = np.append(result_val_mean_np, result_val_mean)
            result_val_max_np = np.append(result_val_max_np, result_val_max)
            result_val_std_np = np.append(result_val_max_np, result_val_std)
            result_val_zero_np = np.append(result_val_zero_np, result_val_zero)
            result_val_max_zero_np = np.append(result_val_max_zero_np, result_val_max_zero)
            epi_len_val_zero_np = np.append(epi_len_val_zero_np, epi_len_val_zero)
        
        
            np.save(filepath+path_logs+'x_step.npy', x_step)
            np.save(filepath+path_logs+'x_episode.npy', x_episode)
            np.save(filepath+path_logs+'one_step_average_reward.npy', one_step_average_reward)
            np.save(filepath+path_logs+'final_step_efficiency.npy', final_step_efficiency)
            np.save(filepath+path_logs+'epsilon_.npy', epsilon_)
            np.save(filepath+path_logs+'max_efficiency.npy', max_efficiency)
            np.save(filepath+path_logs+'memory_size.npy', memory_size)
            np.save(filepath+path_logs+'train_loss.npy', train_loss)
            np.save(filepath+path_logs+'result_val_mean.npy', result_val_mean_np)
            np.save(filepath+path_logs+'result_val_max.npy', result_val_max_np)
            np.save(filepath+path_logs+'result_val_std.npy', result_val_std_np)
            np.save(filepath+path_logs+'result_val_zero.npy', result_val_zero_np)
            np.save(filepath+path_logs+'result_val_max_zero.npy', result_val_max_zero_np)
            np.save(filepath+path_logs+'epi_len_val_zero.npy', epi_len_val_zero_np)

            


            #saved data: hyperparameters(json), logs(csv)
            logger.write_logs(loggername, lgr, sh, n_epi, result_val_max_zero, \
                np.max(result_epi_st), result_flag, epi_length, memory.size(), epsilon*100, count)

            # save_devices
            logger.deviceplotter(filepath+path_devices, s, n_epi)

            # save_np_struct
            logger.numpystructplotter(filepath+path_np_struct, s, n_epi)

            # checkpoint
            torch.save(q.state_dict(), filepath+path_model+str(count)+'steps_q')
            torch.save(q_target.state_dict(), filepath+path_model+str(count)+'steps_q_target')
            

            
        n_epi +=1

    save_summary = True    
    if save_summary == True:
        logger.summaryplotter(q, epi_len_st, s, filepath+path_summary)
    
    np.save(filepath+path_logs+'x_step.npy', x_step)
    np.save(filepath+path_logs+'x_episode.npy', x_episode)
    np.save(filepath+path_logs+'one_step_average_reward.npy', one_step_average_reward)
    np.save(filepath+path_logs+'final_step_efficiency.npy', final_step_efficiency)
    np.save(filepath+path_logs+'epsilon_.npy', epsilon_)
    np.save(filepath+path_logs+'max_efficiency.npy', max_efficiency)
    np.save(filepath+path_logs+'memory_size.npy', memory_size)
    np.save(filepath+path_logs+'train_loss.npy', train_loss)
    np.save(filepath+path_logs+'result_val_mean.npy', result_val_mean_np)
    np.save(filepath+path_logs+'result_val_max.npy', result_val_max_np)
    np.save(filepath+path_logs+'result_val_zero.npy', result_val_zero_np)
    np.save(filepath+path_logs+'result_val_max_zero.npy', result_val_max_zero_np)
    np.save(filepath+path_logs+'epi_len_val_zero.npy', epi_len_val_zero_np)


    final_time = time.process_time()
    
    np.save(filepath+path_logs+'time_elapse.npy', final_time-init_time)
    
    save_model = True
    if save_model == True:
        print('initial eff: {}'.format(result_init))
        print('final eff: {}'.format(result_next))
        print("Qnet's state_dict:")
        for param_tensor in q.state_dict():
            print(param_tensor, "\t", q.state_dict()[param_tensor].size())
        print("Q_target's state_dict:")
        for param_tensor in q_target.state_dict():
            print(param_tensor, "\t", q_target.state_dict()[param_tensor].size())
        print("Qnet's Optimizer's state_dict:")
        for var_name in q.state_dict():
            print(var_name, "\t", q.state_dict()[var_name])

        print("Q_target's Optimizer's state_dict:")
        for var_name in q_target.state_dict():
            print(var_name, "\t", q_target.state_dict()[var_name])

        torch.save(q.state_dict(), filepath+path_model+'final_steps_q')
        torch.save(q_target.state_dict(), filepath+path_model+'final_steps_q_target')

    env.close()
    env_val.close()
    

    print('max efficiency: ', result_flag)
    print('max stepnumber: ', np.load(filepath+path_np_struct_max+'max_stepnumber.npy'))
    print('max strucutre: ', np.load(filepath+path_np_struct_max+'max_structure.npy'))
    print('CPU time: ', final_time - init_time)

