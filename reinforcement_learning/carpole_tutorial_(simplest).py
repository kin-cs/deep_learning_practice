''' tutorial by iambrain from https://gist.github.com/iambrian/2bcc8fc03eaecb2cbe53012d2f505465
	modified and commented by Kin Tse
'''

import gym
env = gym.make('CartPole-v0')  # create the env
highscore = 0  # record the highest score
for _ in range(20):  # run 20 episode
	obs = env.reset()  # get the first obs when start the env
	scores = 0  # keep track of the reward in each episode
	while True:  # keep running until game over
		env.render()
		action = 1 if obs[2] > 0 else 0  # if angle is +ve, move right, otherwise move left
		obs, reward, done, info = env.step(action)
		scores += reward
		if done:
			if scores > highscore:
				highscore = scores
			break

print(highscore)
