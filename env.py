import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces


import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, size = 100):
        super(CustomEnv, self).__init__()

        self.size = size
        self.struct = np.ones(self.size)
        self.result = 0

        #self.action_space = spaces.Discrete(np.prod(self.size))
        #self.observation_space = spaces.Box(low=0, high=1, shape=self.size)
    

    def dbr_simulation(self, layer_indices):
    
        def calc_A(n1, n2, w, c, d):
            speed_of_light = 3e8
            k1 = n1*w/speed_of_light
            k2 = n2*w/speed_of_light
            first_term = np.cos(k2*d)
            second_term = (0.5)*1j*(k2/k1 + k1/k2)*np.sin(k2*d)
            factor = np.exp(1j*k1*c)
            return factor*(first_term+second_term)

        def calc_B(n1, n2, w, c, d):
            speed_of_light = 3e8
            k1 = n1*w/speed_of_light
            k2 = n2*w/speed_of_light
            first_term = (0.5)*1j*(k2/k1 - k1/k2)*np.sin(k2*d)
            factor = np.exp(-1j*k1*c)
            return factor*first_term

        def calc_C(n1, n2, w, c, d):
            speed_of_light = 3e8
            k1 = n1*w/speed_of_light
            k2 = n2*w/speed_of_light
            first_term = -(0.5)*1j*(k2/k1 - k1/k2)*np.sin(k2*d)
            factor = np.exp(1j*k1*c)
            return factor*first_term

        def calc_D(n1, n2, w, c, d):
            speed_of_light = 3e8
            k1 = n1*w/speed_of_light
            k2 = n2*w/speed_of_light
            first_term = np.cos(k2*d)
            second_term = -(0.5)*1j*(k2/k1 + k1/k2)*np.sin(k2*d)
            factor = np.exp(-1j*k1*c)
            return factor*(first_term+second_term)

        def calc_prop_matrix(n1, n2, w, c, d):
            return np.array(
                [
                [calc_A(n1, n2, w, c, d), calc_B(n1, n2, w, c, d)],
                [calc_C(n1, n2, w, c, d), calc_D(n1, n2, w, c, d)]
                ]
            )

        def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        
        mirror_size = 600e-9
        center_wavelength = 400e-9
        num_photons_to_simulate=100
        num_layers = len(layer_indices)
        single_layer_size = mirror_size / num_layers  # size of a single layer in the DBR mirror
        n_high = 2.88  # high refractive index TiO2
        n_low = 1.45  # low refractive index SiO2
        speed_of_light = 3e8
        reflected = []

        # Modify the wavelengths list to only include the center wavelength
        wavelengths = [center_wavelength]

        for lmda in wavelengths:
            w = 2*np.pi*speed_of_light/lmda
            E_field = np.zeros((num_layers+1,2), dtype=np.cfloat)
            E_field[-1] = [1, 0]  # starting E-field

            for layer_idx in reversed(range(num_layers)):
                n1 = n_high if layer_indices[layer_idx] else n_low
                n2 = n_high if layer_indices[layer_idx-1] else n_low if layer_idx > 0 else 1.0  # outside medium
                mat_n1_to_n2 = calc_prop_matrix(n1, n2, w, 0, single_layer_size)
                E_field[layer_idx] = np.dot(mat_n1_to_n2, E_field[layer_idx+1])

            num_of_photons = np.square(np.absolute(E_field))
            num_of_photons /= num_of_photons[0, 0]

            num_reflected = num_of_photons[0][1]
            reflected.append(num_reflected)

        avg_reflected = np.mean(reflected)  # since there is only one element in the reflected list, the mean is the element itself
            
        return avg_reflected
    
    
    def step(self, action):
        done = False
        result_before = self.result
        struct_after= self.struct.copy()
        
        if (struct_after[action] == 1):
            struct_after[action] = 0
        elif(struct_after[action] == 0):
            struct_after[action] = 1
        else:
            raise ValueError('action number cannot exceed cell number')
        
        # Get the reward
        self.result = self.dbr_simulation(self.struct)
        reward = (self.result*100) ** 3

        self.struct = struct_after.copy()


        return struct_after.squeeze(), self.result, reward, done

    def reset(self):
        self.struct = np.ones(self.size)
        result_init = 0
        self.done = False
        
        return self.struct.squeeze(), result_init
    
    def get_obs(self):
        return tuple(self.struct)
    
    def render(self, mode='human'):
        plt.imshow(self.struct.reshape((20,20)), cmap='gray')
        plt.show()
