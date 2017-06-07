'''
Tutorial by Sentdex from https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
Modified and commented by Kin

'''

import gym
import random
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn
from statistics import mean, median
from collections import Counter

import pdb

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():  # testing only
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				break

# some_random_games_first()

def initial_population():
	training_data = []  # observations and the actions we made [s, a]
	scores = []
	accepted_scores = []
	for _ in range(initial_games):  # num of games we play
		score = 0
		game_memory = []  # save the actions
		prev_observation = []
		for _ in range(goal_steps):  # Actual game runs here (# of steps per game)
			# env.render()  # show the action on screen
			action = random.randrange(0, 2)  # only generate 0 or 1
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation = observation
			score += reward
			if done:
				break  # break this for loop

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:  # turn the action (1 or 0) into one-hot [0, 1]
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				training_data.append([data[0], output])  # save all s and a

		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	print('Average accepted score:', mean(accepted_scores))
	print('Median accepted score:', median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data


def neural_net_model(input_size):  # build the NN
	network = input_data(shape = [None, input_size, 1], name='input')  # use input_data to build the model

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)  # the keep rate of Dropout?

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')  # we can change the output to make more categories later
	network = regression(network, optimizer='adam', learning_rate=LR,
		loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')  # save the log in current dir

	return model

def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	Y = [i[1] for i in training_data]

	if not model:
		model = neural_net_model(input_size = len(X[0]))

	model.fit({'input': X}, {'targets': Y}, n_epoch=5, snapshot_step=500, show_metric=True,
			run_id="openaiCartpole")

	return model


training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
		choices.append(action)  # use to check the ratio of diff predictions later

		new_obs, reward, done, info = env.step(action)
		prev_obs = new_obs
		game_memory.append([new_obs, action])
		score += reward
		if done:
			break
	scores.append(score)

print('Average Score', sum(scores)/len(scores))
print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices),
										choices.count(0)/len(choices)))


## saving model
# model.save('carpole.model')
# model.load('carpole.model')































