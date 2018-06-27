import numpy as np

world = [ #															 x
		[ 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' ],# y
		[ 'w' , ' ' ,  3  , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' , ' ' , 'w' , 'w' , 'w' , ' ' , 'w' , 'w' , ' ' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' ,  5  , 'w' ,  4  , ' ' , 'w' , 'e' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , 'w' , -5  , ' ' , 'w' , ' ' , ' ' , 'w' ],
		[ 'w' , -3  , ' ' , ' ' , 'w' , ' ' , ' ' , 'w' ,  9  , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , '@' , 'w' ,  1  ,  6  , 'w' , 'w' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' ,  7  , ' ' , ' ' , ' ' , ' ' , -2  , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' ]
	   ]

class Guy:
	def __init__(self, world='std', finish_value = 10, hunger_rate = 0.05, max_reward = 10, max_steps = 500):
		stdworld = [ #															 x
		[ 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' ],# y
		[ 'w' , ' ' ,  3  , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' , ' ' , 'w' , 'w' , 'w' , ' ' , 'w' , 'w' , ' ' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' ,  5  , 'w' ,  4  , ' ' , 'w' , 'e' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , 'w' , -5  , ' ' , 'w' , ' ' , ' ' , 'w' ],
		[ 'w' , -3  , ' ' , ' ' , 'w' , ' ' , ' ' , 'w' ,  9  , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , '@' , 'w' ,  1  ,  6  , 'w' , 'w' , ' ' , 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' ,  7  , ' ' , ' ' , ' ' , ' ' , -2  , ' ' , ' ' , ' ' , 'w' ],
		[ 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' ]
		]
		if world == 'std':
			self.init_world = self.copy_world(stdworld)
		else:
			self.init_world = self.copy_world(world)
		# self.world = self.init_world[:] #np.array(world)
		# self.x, self.y = self.find_self()
		# self.world[self.x][self.y] = ' '
		self.max_steps = max_steps
		self.hunger_rate = hunger_rate
		self.max_possible = 45
		self.max_reward = max_reward
		self.finish_value = finish_value
		self.movexy = {
			'up':    [-1, 0],
			'right': [ 0, 1],
			'left':  [ 0,-1],
			'down':  [ 1, 0]
		}
		self.reset()
	def copy_world(self, world):
		new_world = []
		for i in world:
			new_world.append(i[:])
		return new_world
	def reset(self):
		self.world = self.copy_world(self.init_world)
		# print(self.world)
		self.world_height = len(self.world   )
		self.world_width = len(self.world[0])
		self.x, self.y = self.find_self()
		self.score = 0
		self.steps = 0
		return self.get_observation()
		# self.world[self.x][self.y] = ' '
	def find_self(self):
		for x in range(     self.world_height ):
			for y in range( self.world_width  ):
				if self.world[x][y] == '@':
					self.world[x][y] = ' '
					# print(x,y)
					return x, y
	def print_world(self):
		print('Score: ', self.score)
		print('Steps: ', self.steps)
		for x in range(len(self.world)):
			line = self.world[x][:]
			for y in range(len(line)):
				if line[y] == 'w':
					line[y] = u"\u2588"*3
				elif self.x == x and self.y == y:
					line[y] = ' @ '
				elif line[y] == ' ':
					line[y] = ' '*3
				elif line[y] == 'e':
					line[y] = ' x '
				elif type(line[y]) == type(0):
					if line[y]>=0:
						line[y] = ' ' +str(line[y])+ ' '
					else:
						line[y] = str(line[y])+ ' '
			print(''.join(line))
	def move(self, where):
		if self.world[self.x + self.movexy[where][0]][self.y + self.movexy[where][1]] == 'w':
			self.steps += 1
		else:
			self.x += self.movexy[where][0]
			self.y += self.movexy[where][1]
			self.steps += 1
		return self.x, self.y
	def act(self, where):
		x,y = self.move(where)
		reward, finish = self.consume_under()
		observation = self.get_observation()
		featurable_reward = reward/self.max_reward
		self.score -= self.hunger_rate
		featurable_score = self.get_featurable_score()
		self.print_world()
		if finish or self.steps > self.max_steps:
			score = self.score
			self.reset()
			return score
		return observation, featurable_reward, featurable_score
		# print(u"\u2588")
	def get_featurable_score(self):
		observe_score = np.zeros((2,))
		if self.score >= 0:
			observe_score[0] =  self.score/self.max_possible
		else:
			observe_score[1] = -self.score/self.max_possible
		return observe_score
	def consume_under(self):
		self.under = self.world[self.x][self.y]
		if self.under == 'e':
			final_reward = self.get_final_reward()
			self.score += final_reward
			return final_reward, True
		elif self.under == ' ':
			return 0, False
		elif type(self.under) == type(1):
			self.score += self.under
			self.world[self.x][self.y] = ' '
			return self.under, False
	def get_final_reward(self):
		return self.finish_value/self.steps
	def get_observation(self):
		image = np.zeros((self.world_height, self.world_width, 4))
		for x in range(self.world_width):
			for y in range(self.world_height):
				if self.world[x][y] == 'e':
					image[x][y] = (0, 1, 1, 0) #(wall, exit, value) #(1-1/self.steps, 1/self./self.max_possible, )
				elif self.world[x][y] == 'w':
					image[x][y] = (1, 0, 0, 0)
				elif type(self.world[x][y]) == type(1):
					if self.world[x][y] >=0:
						image[x][y] = (0, 0, self.world[x][y]/self.max_reward, 0)
					else:
						image[x][y] = (0, 0, 0, -self.world[x][y]/self.max_reward)
				else:
					image[x][y] = (0, 0, 0, 0)
		return image

# a = Guy(world, finish_value = 100, hunger_rate = 0.05)

# o, r, s = a.act('up')


