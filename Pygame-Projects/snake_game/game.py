import pygame
import sys
from pygame.math import Vector2
from food import Food
from snake import Snake

# Definitions --------------------------------------
pygame.init()
#Colors for the game
GREEN = (173,204,96)
DARK_GREEN = (43, 51, 24)
RED = (255,0,0)
# Grid system for the game
cell_size = 30
cells_number = 25
# Declaration of the size of the window
screen = pygame.display.set_mode((cell_size*cells_number,cell_size*cells_number))
#Definition of a game class
class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food(self.snake.body)
    def draw(self):
        self.snake.draw(cell_size,screen,DARK_GREEN)
        self.food.draw(cell_size,screen,RED)
    def update(self):
        self.snake.update()
        self.check_collision_with_food()
    def check_collision_with_food(self):
        if self.snake.body[0] == self.food.position:
            self.food.position = self.food.generate_random_pos(self.snake.body)
            self.snake.add_segment = True
# Title of the game, visible at the top of the window
pygame.display.set_caption("Retro snake")
# Declaration of the clock object
clock = pygame.time.Clock()
#Creation of the game object
game = Game()
#Specific event for the movement of the snake
SNAKE_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SNAKE_UPDATE,200)

#---------------------------------------------------

# Game Loop ----------------------------------------
while True:
    for event in pygame.event.get():
        if event.type == SNAKE_UPDATE:
            game.update()
        if event.type == pygame.QUIT:
            #Close the game when pressing the x button in the window
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and game.snake.direction != Vector2(0,1):
                game.snake.direction = Vector2(0,-1)
            if event.key == pygame.K_DOWN and game.snake.direction != Vector2(0,-1):
                game.snake.direction = Vector2(0,1)
            if event.key == pygame.K_LEFT and game.snake.direction != Vector2(1,0):
                game.snake.direction = Vector2(-1,0)
            if event.key == pygame.K_RIGHT and game.snake.direction != Vector2(-1,0):
                game.snake.direction = Vector2(1,0)

    #Fill the screen with the GREEN color
    screen.fill(GREEN)
    #Draw of the game elemets
    game.draw()
    # Update of the game
    pygame.display.update()
    #Definition of the fps (frames per second for the game)
    clock.tick(60)
#---------------------------------------------------