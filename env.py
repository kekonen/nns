import numpy as np

world = [ #															 x
		[ 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w', 'w', 'w' ],# y
		[ 'w' , ' ' ,  3  , ' ' , ' ' , ' ' , ' ' , ' ' , ' ', ' ', 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ', ' ', 'w' ],
		[ 'w' , ' ' , 'w' , 'w' , 'w' , ' ' , 'w' , 'w' , ' ', ' ', 'w' ],
		[ 'w' , ' ' , ' ' ,  5  , 'w' ,  4  , ' ' , 'w' ,  0 , ' ', 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , 'w' , -5  , ' ' , 'w' , ' ', ' ', 'w' ],
		[ 'w' , -3  , ' ' , ' ' , 'w' , ' ' , ' ' , 'w' ,  9 , ' ', 'w' ],
		[ 'w' , ' ' , ' ' , '@' , 'w' ,  1  ,  6  , 'w' , 'w', ' ', 'w' ],
		[ 'w' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ' , ' ', ' ', 'w' ],
		[ 'w' ,  7  , ' ' , ' ' , ' ' , ' ' , -2  , ' ' , ' ', ' ', 'w' ],
		[ 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w' , 'w', 'w', 'w' ]
	   ]

class Guy:
	def __init__(self, world, cool_value, hunger_rate):
		self.world = world #np.array(world)
		self.world_height = len(self.world   )
		self.world_width = len(self.world[0])
		self.x, self.y = self.find_self()
		self.world[self.x][self.y] = ' '
		self.hunger_rate = hunger_rate
		self.cool_value = cool_value
		self.score = 0
		self.steps = 0
		self.movexy = {
			'up':    [-1, 0],
			'right': [ 0, 1],
			'left':  [ 0,-1],
			'down':  [ 1, 0]
		}
	def find_self(self):
		for x in range(     self.world_height ):
			for y in range( self.world_width  ):
				if self.world[x][y] == '@':
					return x, y
	def print_world(self):
		for x in range(len(self.world)):
			line = self.world[x]
			for y in range(len(line)):
				if line[y] == 'w':
					line[y] = u"\u2588"*3
				elif self.x == x and self.y == y:
					line[y] = ' @ '
				elif line[y] == ' ':
					line[y] = ' '*3
				elif line[y] == 0:
					line[y] = ' x '
				elif type(line[y]) == type(0):
					if line[y]>=0:
						line[y] = ' ' +str(line[y])+ ' '
					else:
						line[y] = str(line[y])+ ' '
			print(''.join(line))
	def move(self, where):
		if self.world[self.x + movexy[where][0]][self.y + movexy[where][1]] == 'w':
			self.steps += 1
		else:
			self.x += movexy[where][0]
			self.y += movexy[where][1]
			self.steps += 1
		return self.x, self.y
	def act(self, where):
		x,y = self.move(where)
		reward, finish = self.consume_under()
		observation = self.get_observation()
		self.score -= self.hunger_rate
		print(u"\u2588")
	def consume_under(self):
		self.under = self.world[self.x][self.y]
		if self.under == 0:
			return self.cool_value/self.steps, 
		elif self.under == ' ':
			return 0
		elif type(self.under) == type(1):
			self.score += self.under
			self.world[self.x][self.y] = ' '
			return self.under
	def get_observation(self,where):
		image = np.zeros((self.world_hight, self.world_width, 4))
		for x in range(len(self.world)):
			for y in range(len(self.world[0])):
				if self.world[x][y] == 0:
					image[x][y] = (0, 1, 1) #(wall, exit, value) #(1-1/self.steps, 1/self./self.max_possible, )
				elif self.world[x][y] == 'w':
					image[x][y] = (1, 0, 0)
				elif type(self.world[x][y]) == type(1):
					image[x][y] = (0, 0, self.world[x][y]/self.max_possible)
				else:
					image[x][y] = (0, 0, 0)

a = Guy(world, 1, 1)





