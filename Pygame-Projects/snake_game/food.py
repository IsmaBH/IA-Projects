import pygame
from pygame.math import Vector2
import random

class Food:
    def __init__(self,snake_body):
        self.position = self.generate_random_pos(snake_body)
    
    def draw(self, cell_size, surface, color):
        #The Rect() function recives x,y,width,height parameters
        food_rect = pygame.Rect(self.position.x * cell_size, self.position.y * cell_size,cell_size,cell_size)
        pygame.draw.rect(surface, color, food_rect)

    def generate_random_cell(self):
        x = random.randint(0,24)
        y = random.randint(0,24)
        return Vector2(x,y)
    
    def generate_random_pos(self,snake_body):
        position = self.generate_random_cell()
        while position in snake_body:
            position = self.generate_random_cell()
        return position