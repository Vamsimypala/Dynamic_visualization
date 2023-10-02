import pygame
import random
import sys

pygame.init()

WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 20
BALL_RADIUS = 15
BALL_SPEED = 5
PADDLE_SPEED = 7
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Ball")

# Create the paddle
paddle = pygame.rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT)

# Initialize ball variables
balls = []
ball_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
ball_speeds = [BALL_SPEED, BALL_SPEED + 1, BALL_SPEED + 2]

# Function to create a new ball
def create_ball():
    color = random.choice(ball_colors)
    speed = random.choice(ball_speeds)
    return pygame.Rect(random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS),
                       0, BALL_RADIUS * 2, BALL_RADIUS * 2), color, speed

# Main game loop
clock = pygame.time.Clock()
score = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Move the paddle
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.left -= PADDLE_SPEED
    if keys[pygame.K_RIGHT] and paddle.right < WIDTH:
        paddle.left += PADDLE_SPEED

    # Create new balls at regular intervals
    if random.randint(0, 100) < 3:
        balls.append(create_ball())

    # Move the balls
    for ball, color, speed in balls:
        ball.move_ip(0, speed)

    # Check for collisions with the paddle
    for ball, color, speed in balls[:]:
        if ball.colliderect(paddle):
            balls.remove((ball, color, speed))
            score += 1

    # Remove balls that have gone off the screen
    for ball, color, speed in balls[:]:
        if ball.top > HEIGHT:
            balls.remove((ball, color, speed))

    # Clear the screen
    screen.fill(BLACK)

    # Draw the paddle
    pygame.draw.rect(screen, WHITE, paddle)

    # Draw the balls
    for ball, color, speed in balls:
        pygame.draw.circle(screen, color, ball.center, BALL_RADIUS)

    # Draw the score
    font = pygame.font.Font(None, 36)
    score_text = font.render("Score: " + str(score), True, WHITE)
    screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(60)
