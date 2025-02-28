import pygame
from pygame.math import Vector2


class Snake:
    def __init__(self):
        self.body = [Vector2(6,9),Vector2(5,9),Vector2(4,9)]
        self.direction = Vector2(1,0)
        self.add_segment = False

    def draw(self,cell_size,surface,color,offset):
        for segment in self.body:
            segment_rect = (offset+segment.x*cell_size,offset+segment.y*cell_size,cell_size,cell_size)
            pygame.draw.rect(surface,color,segment_rect,0,7)

    def update(self):
        if self.add_segment:
            self.body.insert(0,self.body[0]+self.direction)
            self.add_segment = False
        else:
            self.body = self.body[:-1]
            self.body.insert(0,self.body[0]+self.direction)

    def reset(self):
        self.body = [Vector2(6,9),Vector2(5,9),Vector2(4,9)]
        self.direction = Vector2(1,0)