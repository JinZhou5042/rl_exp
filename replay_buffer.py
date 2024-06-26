import random
import warnings
from collections import deque
MINIBATCH_SIZE = 64

class Buffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen=self.limit)
        
    def __len__(self):
        return len(self.data)
    
    def sample_batch(self, batchSize):
        if len(self.data) < batchSize:
            warnings.warn('Not enough entries to sample without replacement.')
            return None
        else:
            batch = random.sample(self.data, batchSize)
            current_state = [element[0] for element in batch]
            action = [element[1] for element in batch]
            next_state = [element[2] for element in batch]
            reward = [element[3] for element in batch]
            terminal = [element[4] for element in batch]
        return current_state, action, next_state, reward, terminal
                  
    def append(self, element):
        self.data.append(element)  
