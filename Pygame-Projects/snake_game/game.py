import pygame
import sys
from pygame.math import Vector2
from food import Food
from snake import Snake

# Definitions --------------------------------------
pygame.init()
title_font = pygame.font.Font(None,60)
score_font = pygame.font.Font(None,60)
#Colors for the game
GREEN = (173,204,96)
DARK_GREEN = (43, 51, 24)
RED = (255,0,0)
# Grid system for the game
cell_size = 30
cells_number = 25
#Values to give the game window a margin of 75 pixels and a border to the game area
OFFSET = 75
# Declaration of the size of the window
screen = pygame.display.set_mode((2*OFFSET + cell_size*cells_number,2*OFFSET + cell_size*cells_number))

#Definition of a game class
class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.state = "RUNNING"
        self.score = 0
    def draw(self):
        self.snake.draw(cell_size,screen,DARK_GREEN,OFFSET)
        self.food.draw(cell_size,screen,RED,OFFSET)
    def update(self):
        if self.state == "RUNNING":
            self.snake.update()
            self.check_collision_with_food()
            self.check_collision_with_edges()
            self.check_collition_with_tail()
    def check_collision_with_food(self):
        if self.snake.body[0] == self.food.position:
            self.food.position = self.food.generate_random_pos(self.snake.body)
            self.snake.add_segment = True
            self.score += 1
    def check_collision_with_edges(self):
        if self.snake.body[0].x == cells_number or self.snake.body[0].x == -1:
            self.game_over()
        if self.snake.body[0].y == cells_number or self.snake.body[0].y == -1:
            self.game_over()
    def check_collition_with_tail(self):
        headless_body  = self.snake.body[1:]
        if self.snake.body[0] in headless_body:
            self.game_over()
    def game_over(self):
        self.snake.reset()
        self.food.position = self.food.generate_random_pos(self.snake.body)
        self.state = "STOPPED"
        self.score = 0
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
            if game.state == "STOPPED":
                game.state = "RUNNING"
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
    pygame.draw.rect(screen,DARK_GREEN,(OFFSET-5,OFFSET-5,cell_size*cells_number+10,cell_size*cells_number+10),5)
    #Draw of the game elemets
    game.draw()
    title_surface = title_font.render("Retro snake", True,DARK_GREEN)
    score_surface = score_font.render("Score: {}".format(str(game.score)),True,DARK_GREEN)
    screen.blit(title_surface,(OFFSET-5,20))
    screen.blit(score_surface,(OFFSET-5,OFFSET+cell_size*cells_number+10))
    # Update of the game
    pygame.display.update()
    #Definition of the fps (frames per second for the game)
    clock.tick(60)
#---------------------------------------------------