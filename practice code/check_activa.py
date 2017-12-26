
import gym
import ppaquette_gym_super_mario
import random

import numpy as np
import gym 
from wrapper import action_space
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input, Lambda, merge
from keras.optimizers import RMSprop
from keras.models import Model
#from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np

from sum_tree import SumTree
import random


env = gym.make('ppaquette/meta-SuperMarioBros-v0')

wra_act=action_space

#reduce actions
env=wra_act.mario_action(env)

#reduce pixel


env=wra_act.ProcessFrame84(env)
actions=(
    [0, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 1, 0, 0, 0, 0],  [0, 1, 0, 0, 1, 0],  [0, 1, 0, 0, 0, 1],[0, 1, 0, 0, 1, 1], 
    [0, 0, 0, 1, 0, 0],[0, 0, 0, 1, 1, 0],  
    [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1],  
     [0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1], 
     [0, 0, 0, 0, 1, 1])


import random
action_size=6
epsilon = 1


def append_sample(history, action, reward, next_history, done):
    #TD ERROR 
    target=model.predict([history])
    old_val = target[0][action]
    target_val = target_model.predict(([next_history]))
    
    if done:
        target[0][action] = reward
    else:
        target[0][action] = reward + discount_factor *  (np.amax(target_val[0]))
    error= abs(old_val - target[0][action1])
    memory.add(error,(history,action, reward, next_history, done))


def train_model():
    if epsilon > epsilon_end:
        epsilon -= epsilon_decay_step
    mini_batch = memory.sample(batch_size)
    
    errors= np.zeros(batch_size)
    history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
    next_history = np.zeros((self.batch_size, self.state_size[0],
                             self.state_size[1], self.state_size[2]))
    target = np.zeros((self.batch_size,))
    action, reward, dead = [], [], []

    for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][1][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][1][3] / 255.)
            action.append(mini_batch[i][1][1])
            reward.append(mini_batch[i][1][2])
            dead.append(mini_batch[i][1][4])

    curr_q = self.model.predict(history)
    value = self.model.predict(next_history)
    target_value = self.target_model.predict(next_history)
    for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]
            errors[i] = abs(curr_q[i][action[i]] - target[i])

        # TD-error로 priority 업데이트
    for i in range(self.batch_size):
            idx = mini_batch[i][0]
            self.memory.update(idx, errors[i])

    loss = optimizer([history, action, target])
    avg_loss += loss[0]
    
    
def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train


class Memory:
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)





def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())



epsilon=0.99
def get_action():
    

    
    if np.random.rand() <= epsilon:
        return random.choice(actions)
    else:
        q_value = model.prediction(history)
        return np.argmax(q_value[0])
    

class DQNAgent:
    def __init__(self, action_size, env):
        self.env = env
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 400000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = Memory(400000)
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.avg_q_max, self.avg_loss = 0,0

env = env
agent = DQNAgent(action_size=6, env=env)
Memory(400000)
scores, episodes, global_step = [], [], 0

for e in range(100):
        done = False
        max_x = 0
        now_x = 0
        hold_frame = 0
        before_max_x = 200

        start_position = 500
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        state = observe
        history = np.stack((state, state, state, state), axis=2)
    #    history = np.reshape([history], (1, 84, 84, 4))

        action_count = 0
        action = get_action()

        while not done:
            global_step += 1
            step += 1



            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, clear = \
                env.step(action)

           # if now_x >= 8776:
              #  reward = 300
              #  done = True

           #if done and now_x < 8776:
              # reward = -100

          #  reward /= 100
            # reward = np.clip(reward, -1., 1.)

            # 각 타임스텝마다 상태 전처리
            next_state = observe
           # next_state = np.reshape([next_state], (1, 84, 84, 1))
           # next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, done)

            if global_step >= agent.train_start:
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward
            history = next_history

            if now_x <= before_max_x:
                hold_frame += 1
                if hold_frame > 2000:
                    break
            else:
                hold_frame = 0
                before_max_x = max_x

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", score, "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
       # if e % 1000 == 0:
        #    agent.model.save_weights("./save_model/supermario_per.h5")



