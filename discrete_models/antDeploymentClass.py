import pygame
import random
import math
# from queue import PriorityQueue

WIDTH = 800
# HEIGHT = 100

RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GREY = (128, 128, 128)

class Node:
	def __init__(self, row, col, width, height, total_rows, full_nest):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * height
		if full_nest:
			self.colour = RED
		else:
			self.colour = WHITE
		self.width = width
		self.height = height
		#self.height = height
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	def is_ant(self):
		return self.colour == RED

	def reset(self):
		return self.colour == WHITE

	def make_ant(self):
		self.colour = RED

	def remove_ant(self):
		self.colour = WHITE

	def draw(self, win):
		pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.height))


class AntSelection:

	def __init__(self, rows, cols, full_nest=False, num_ants=70, rand=True):

		self.WIDTH = rows * 25
		self.HEIGHT = cols * 25
		self.selected_ants = []
		self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) # For this, will use the nest heigh/depth
		self.rows = rows
		self.cols = cols
		self.full_nest = full_nest
		self.num_ants = num_ants
		self.rand = rand
		if not full_nest:
			pygame.display.set_caption("Selecting position of ants")
		else:
			pygame.display.set_caption("Simulation composition")


	def make_random_positions(self):
		all_positions = set([(x,y) for x in range(self.rows) for y in range(self.cols)])
		ant_positions = set(random.sample(all_positions,self.num_ants))
		empty_cells = all_positions.difference(ant_positions)
		return ant_positions, empty_cells

	def make_grid(self, rows, cols, width, height):
		grid = []
		gap= width // rows
		hgap = height // cols
		if self.rand:
			ant_positions, empty_cells = self.make_random_positions()
			for i in range(rows):
				row = []
				for j in range(cols):
					if (i,j) in ant_positions:
						row.append(Node(i,j,gap,hgap,rows,True))
					else:
						row.append(Node(i,j,gap,hgap,rows,False))
				grid.append(row)
		else:
			for i in range(rows):
				grid.append([Node(i, j, gap, hgap, rows, self.full_nest) for j in range(cols)])
				#grid.append([]))
				#for j in range(cols):
				#	spot = Node(i, j, gap, gap, rows)
				#	grid[i].append(spot)

		return grid


	def draw_grid(self, win, rows, cols, width, height):
		gap = width //rows
		hgap = height // cols
		for i in range(cols):
			pygame.draw.line(win, GREY, (0, i * hgap), (width, i * hgap))
			for j in range(rows):
				pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


	def draw(self, win, grid, rows, cols, width, height):
		win.fill(WHITE)

		for row in grid:
			for node in row:
				node.draw(win)

		self.draw_grid(win, rows, cols, width, height)
		pygame.display.update()


	def get_clicked_pos(self, pos, rows, cols, width, height):
		gap = width // rows
		hgap = height // cols
		y, x = pos

		row = y // gap
		col = x // hgap

		return row, col

	def main(self):
		ROWS = self.rows
		COLS = self.cols

		grid = self.make_grid(ROWS, COLS,  self.WIDTH, self.HEIGHT)

		run = True
		while run:
			self.draw(self.win, grid, ROWS, COLS, self.WIDTH, self.HEIGHT)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False

				if pygame.mouse.get_pressed()[0]:
					pos = pygame.mouse.get_pos()
					row, col = self.get_clicked_pos(pos, ROWS, COLS, self.WIDTH, self.HEIGHT)
					node = grid[row][col]

					node.make_ant()

				elif pygame.mouse.get_pressed()[2]:
					pos = pygame.mouse.get_pos()
					row, col = self.get_clicked_pos(pos, ROWS, COLS, self.WIDTH, self.HEIGHT)
					node = grid[row][col]

					node.remove_ant()

				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RETURN:
						run = False

		for row in grid:
			for node in row:
				if node.is_ant():
					x = node.row +1
					y = node.col
					self.selected_ants.append([x, y])

		pygame.quit()



