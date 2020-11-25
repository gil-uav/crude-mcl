import random
import time

import cv2
import numpy as np
import pygame

# GLOBAL STATIC VARIABLES
MAP = cv2.imread("data/reference_gray.png", 0)
# x,y are inverted
MAP_SIZE = [MAP.shape[1], MAP.shape[0]]

HEIGHT_PIXELS = 700  # Roughly 120m in map
PIXEL_HEIGHT_RELATION = 120 / HEIGHT_PIXELS
MAX_HEIGHT_PIXELS = 150 / PIXEL_HEIGHT_RELATION
MIN_HEIGHT_PIXELS = 80 / PIXEL_HEIGHT_RELATION

MAX_PARTICLES = 2 ** 13
MIN_PARTICLES = 2 ** 9
NO_PARTICLES = MAX_PARTICLES
NO_SAMPLES = 256
SEARCH_SPACE = 1000

MOVEMENT_NOISE = 3
HEADING_NOISE = 0.1
ELEVATION_NOISE = 0.2
SENSE_NOISE = 0.2
MAX_PROB = np.exp(-(0 ** 2) / (SENSE_NOISE ** 2) / 2.0) / np.sqrt(
    2.0 * np.pi * (SENSE_NOISE ** 2)
)

# UAV starting point
ORIGIN = (random.randrange(MAP_SIZE[0]), random.randrange(MAP_SIZE[1]))
ORIENTATION = random.randrange(360)
# UAV movement
TURN_RATE = -0.01
MOVEMENT_RATE = 10.0
ASCENT_RATE = -2

# Pygame
BACKGROUND = pygame.image.load("data/reference_low.png")
GAME_DIMENSIONS = (1440, 1080)  # With map relation
GAME_MAP_SIZE = MAP_SIZE[0] / GAME_DIMENSIONS[0]
TICK_RATE = 10000
SCREEN = pygame.display.set_mode((GAME_DIMENSIONS[0], GAME_DIMENSIONS[1]))
pygame.display.set_caption("Monte Carlo Localization")
CLOCK = pygame.time.Clock()

# Driving and turning speed
DELTA_ORIENT = -0.007
DELTA_FORWARD = 10
DELTA_ELEVATION = 0.0

RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 155, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def init_particles(
    x_bound, y_bound, z_bound, no_particles: int = 2e5, uniform=True, scale=0.0
):
    """
    Initialize particles.
    """
    if uniform:
        x = np.round(np.random.uniform(x_bound[0], x_bound[1], no_particles))
        y = np.round(np.random.uniform(y_bound[0], y_bound[1], no_particles))
        z = np.round(np.random.uniform(z_bound[0], z_bound[1], no_particles))
    else:
        _x = x_bound[0] + ((x_bound[1] - x_bound[0]) // 2)
        _y = y_bound[0] + ((y_bound[1] - y_bound[0]) // 2)
        x = np.clip(np.round(np.random.normal(_x, scale, no_particles)), 0, MAP_SIZE[0])
        y = np.clip(np.round(np.random.normal(_y, scale, no_particles)), 0, MAP_SIZE[1])
        z = np.round(np.random.normal(z_bound[0], z_bound[1], no_particles))
    theta = np.random.random_sample((no_particles,)) * 2.0 * np.pi
    prob = np.zeros(no_particles)

    particles = np.array([x, y, z, theta, prob]).T

    return particles


def move_particles(particles, turn_rate, movement_rate, ascent_rate):
    """
    Append movement to particles.
    """
    n = len(particles)

    # Orientation
    particles[:, 3] += turn_rate + np.random.normal(0.0, HEADING_NOISE, n)
    particles[:, 3] %= 2 * np.pi

    delta_move = movement_rate * np.random.normal(0.0, MOVEMENT_NOISE, n)
    delta_elevation = ascent_rate * np.random.normal(0.0, MOVEMENT_NOISE, n)

    # X and Y movement
    particles[:, 0] += (delta_move * np.cos(particles[:, 3])) + np.random.normal(
        0.0, MOVEMENT_NOISE, n
    )
    particles[:, 1] -= (delta_move * np.sin(particles[:, 3])) + np.random.normal(
        0.0, MOVEMENT_NOISE, n
    )
    particles[:, 2] += delta_elevation
    np.clip(
        particles[:, 2],
        MIN_HEIGHT_PIXELS - 1,
        MAX_HEIGHT_PIXELS + 1,
        out=particles[:, 2],
    )

    # Cyclic truncate
    particles[:, 0] %= MAP_SIZE[0]
    particles[:, 1] %= MAP_SIZE[1]

    return particles


def move_uav(uav, turn_rate, movement_rate, ascent_rate):
    """
    Appends movement to the UAV(Robot)
    """
    # Orientation
    uav[3] = uav[3] + turn_rate
    uav[3] %= 2 * np.pi

    # X and Y movement
    uav[0] += movement_rate * np.cos(uav[3])
    uav[1] -= movement_rate * np.sin(uav[3])
    uav[2] += ascent_rate

    # Cyclic truncate
    uav[0] %= MAP_SIZE[0]
    uav[1] %= MAP_SIZE[1]

    return uav


def sense(particles, samples):
    """
    Senses the map below(Particels)
    """
    c = np.cos(particles[:, 3])
    s = np.sin(particles[:, 3])
    x = particles[:, 0]
    y = particles[:, 1]
    z = particles[:, 2] // 2
    sx = samples[0]
    sy = samples[1]
    px = np.clip(
        (x + z + c * (np.subtract(sx, z)) - s * (np.subtract(sy, z))).astype(int),
        0,
        MAP_SIZE[0] - 1,
    )
    py = np.clip(
        (y + z + s * (np.subtract(sx, z)) + c * (np.subtract(sy, z))).astype(int),
        0,
        MAP_SIZE[1] - 1,
    )
    pixels = MAP[py, px].T

    return pixels


def sense_uav(particle, samples):
    """
    Senses the map below(UAV)
    """
    c = np.cos(particle[3])
    s = np.sin(particle[3])
    x = particle[0]
    y = particle[1]
    z = particle[2] // 2
    sx = samples[0]
    sy = samples[1]
    px = np.clip((x + z + c * (sx - z) - s * (sy - z)).astype(int), 0, MAP_SIZE[0] - 1)
    py = np.clip((y + z + s * (sx - z) + c * (sy - z)).astype(int), 0, MAP_SIZE[1] - 1)

    return MAP[py, px]


def generate_samples(no_samples, normal=True):
    """
    Pixel coordinates for sensing.
    """
    if normal:
        sx = np.clip(
            np.round(np.random.normal(loc=350, scale=130, size=no_samples)),
            0,
            MAX_HEIGHT_PIXELS,
        )
        sy = np.clip(
            np.round(np.random.normal(loc=350, scale=130, size=no_samples)),
            0,
            MAX_HEIGHT_PIXELS,
        )
    else:
        sx = np.random.randint(0, MAX_HEIGHT_PIXELS, no_samples)
        sy = np.random.randint(0, MAX_HEIGHT_PIXELS, no_samples)

    return sx, sy


def measure_probability(measurement, pixels):
    """
    Performs sense-comparison of particle and UAV.
    """
    equal = np.equal(pixels, measurement)
    not_equal = np.invert(equal)
    not_void = pixels < 250
    void = np.invert(not_void)

    true_positive = np.sum(np.where(not_void & equal, 1, 0), axis=1)
    false_positive = np.sum(np.where(not_void & not_equal, 1, 0), axis=1)
    false_negative = np.sum(np.where(void & not_equal, 1, 0), axis=1)
    precision = np.where(
        (true_positive != 0) & (false_positive != 0),
        (true_positive / (true_positive + false_positive)),
        0,
    )
    recall = np.where(
        (true_positive != 0) & (false_negative != 0),
        (true_positive / (true_positive + false_negative)),
        0,
    )
    f1 = np.where(
        (precision != 0) & (recall != 0),
        2 * ((precision * recall) / (precision + recall)),
        0,
    )
    error = 1.0 - f1

    # To percentage
    probability = (
        np.exp(-0.5 * ((error - 0.0) / SENSE_NOISE) ** 2)
        / (SENSE_NOISE * np.sqrt(2 * np.pi))
        / MAX_PROB
    ) * 100

    return probability


def draw_uav(uav):
    """
    Draws UAV on screen.
    """
    x = int(uav[0] / GAME_MAP_SIZE)
    y = int(uav[1] / GAME_MAP_SIZE)
    orientation = uav[3]
    SCREEN.blit(BACKGROUND, (0, 0))
    w = int(HEIGHT_PIXELS / GAME_MAP_SIZE) // 2
    pygame.draw.circle(SCREEN, pygame.Color(255, 0, 0, 100), (int(x), int(y)), w)
    line_length = 50
    pygame.draw.line(
        SCREEN,
        BLACK,
        (x, y),
        (
            int(x + line_length * np.cos(orientation)),
            int(y - line_length * np.sin(orientation)),
        ),
        4,
    )


def draw_particles(particles):
    """
    Draws particles on screen.
    """
    for p in particles:
        x = p[0] / GAME_MAP_SIZE
        y = p[1] / GAME_MAP_SIZE
        orientation = p[3]
        line_length = 7
        pygame.draw.line(
            SCREEN,
            BLACK,
            (int(x), int(y)),
            (
                int(x + line_length * np.cos(orientation)),
                int(y - line_length * np.sin(orientation)),
            ),
            2,
        )
        circle_color = (0, int(200 * p[4] / 100), 0)
        circle_size = int(1 + (8 * p[4] / 100))
        pygame.draw.circle(SCREEN, circle_color, (int(x), int(y)), circle_size)


def predict_position(particles, segment):
    """
    Predicts UAV position.
    """
    sort_seg = particles[particles[:, 4].argsort()][
        int(len(particles) * (1 - segment)) : :
    ]
    x = np.mean(sort_seg[:, 0])
    y = np.mean(sort_seg[:, 1])
    z = np.mean(sort_seg[:, 2])

    return [x, y, z]


def evaluate(uav, particles):
    """
    Evaluates particle and uav comparison.
    """
    x = uav[0]
    y = uav[1]
    error = (
        np.sum(np.sqrt((x - particles[:, 0]) ** 2 + (y - particles[:, 1]) ** 2))
        / NO_PARTICLES
    )

    return error


def main():
    """
    Main loop.
    """
    global NO_PARTICLES, ASCENT_RATE, TURN_RATE, TICK_RATE
    # Simulate UAV
    origin = (random.randrange(MAP_SIZE[0]), random.randrange(MAP_SIZE[1]))
    orientation = random.randrange(360)
    uav = np.array([origin[0], origin[1], MAX_HEIGHT_PIXELS, np.deg2rad(orientation)]).T

    # Create particles
    particles = init_particles(
        (0, MAP_SIZE[0]),
        (0, MAP_SIZE[1]),
        (MIN_HEIGHT_PIXELS, MAX_HEIGHT_PIXELS),
        NO_PARTICLES,
    )
    particles[:, 3] = uav[3]

    # Simulation state
    exit = False
    pause = False

    step = 0
    # Simulation loop

    while not exit:

        while pause:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pause = False

        start = time.time()
        step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = True
                elif event.key == pygame.K_r:
                    main()

        if step % 100 == 0:
            TURN_RATE = -TURN_RATE
            ASCENT_RATE = -ASCENT_RATE

        # Update particles
        # particles[:, 3] = uav[3]
        particles[:, 2] = uav[2]
        particles = move_particles(particles, TURN_RATE, MOVEMENT_RATE, ASCENT_RATE)
        uav = move_uav(uav, TURN_RATE, MOVEMENT_RATE, ASCENT_RATE)

        predicted_position = predict_position(particles, segment=0.01)
        error = evaluate(uav, particles)
        offset = [
            abs(uav[0] - predicted_position[0]),
            abs(uav[1] - predicted_position[1]),
            abs(uav[2] - predicted_position[2]),
        ]

        samples = generate_samples(NO_SAMPLES, normal=False)

        uav_pixels = sense_uav(uav, samples)
        sx = np.repeat([samples[0]], NO_PARTICLES, axis=0).T
        sy = np.repeat([samples[1]], NO_PARTICLES, axis=0).T

        pixels = sense(particles, (sx, sy))
        particles[:, 4] = measure_probability(uav_pixels, pixels)

        weights = particles[:, 4]
        p = np.zeros_like(particles)
        index = int(np.random.random_sample() * NO_PARTICLES)
        beta = 0.0
        mw = max(weights)

        for i in range(NO_PARTICLES):
            beta += np.random.random_sample() * 2.0 * mw

            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % NO_PARTICLES
            p[i] = particles[index]
        particles = np.copy(p)

        x_std = np.std(particles[:, 0])
        y_std = np.std(particles[:, 1])
        x_std_norm = x_std / (MAP_SIZE[0] // 2)
        y_std_norm = y_std / (MAP_SIZE[1] // 2)
        avg_std = np.mean([x_std, y_std]) / (np.mean([MAP_SIZE[0], MAP_SIZE[1]]) / 2)

        # Resample worst particles inside reduced search space

        if avg_std < 0.3:
            no_resample = int(NO_PARTICLES / 2)
            rss_x = (
                int(predicted_position[0] - SEARCH_SPACE),
                int(predicted_position[0] + SEARCH_SPACE),
            )
            rss_y = (
                int(predicted_position[1] - SEARCH_SPACE),
                int(predicted_position[1] + SEARCH_SPACE),
            )
            rss_z = (MIN_HEIGHT_PIXELS, MAX_HEIGHT_PIXELS)
            particles[:no_resample:] = init_particles(
                rss_x, rss_y, rss_z, no_resample, uniform=False, scale=250
            )
            particles[:no_resample:, 3] = uav[3]
            particles[:no_resample:, 2] = uav[2]
            particles[:no_resample:] = move_particles(
                particles[:no_resample:], TURN_RATE, MOVEMENT_RATE, ASCENT_RATE
            )

        particles = particles[particles[:, 4].argsort()]
        prob = particles[-1, 4]
        prob_avg = np.mean(particles[:, 4])

        # Resample all particles if no probable location found
        if prob < 3:
            NO_PARTICLES = MAX_PARTICLES
            particles = init_particles(
                (0, MAP_SIZE[0]),
                (0, MAP_SIZE[1]),
                (MIN_HEIGHT_PIXELS, MAX_HEIGHT_PIXELS),
                NO_PARTICLES,
            )
            particles[:, 3] = uav[3]

        # Adjust number of particles based on average standard deviation
        if prob_avg > 5:
            NO_PARTICLES = int(
                np.clip(NO_PARTICLES * avg_std, MIN_PARTICLES, MAX_PARTICLES)
            )
            particles = particles[particles[:, 4].argsort()][
                int(len(particles) - NO_PARTICLES) : :
            ]
            particles[:, 3] = uav[3]

        SCREEN.fill(WHITE)

        draw_uav(uav)
        draw_particles(particles)

        pygame.display.update()
        SCREEN.blit(BACKGROUND, [0, 0])
        CLOCK.tick(TICK_RATE)

        elapsed = time.time() - start
        print("Step[{}] | ".format(step), end="")
        print("#nP[{}] | ".format(NO_PARTICLES), end="")
        print(
            "Pred[{:.0f},{:.0f}, {:.0f}] | ".format(
                predicted_position[0], predicted_position[1], predicted_position[2]
            ),
            end="",
        )
        print("Prob[{:.2f}%] | ".format(prob), end="")
        print("AvgProb[{:.2f}%] | ".format(prob_avg), end="")
        print(
            "Std[{:.2f}, {:.2f}, avg: {:.2f}] |".format(
                x_std_norm, y_std_norm, avg_std
            ),
            end="",
        )
        print("UAV[{:.0f},{:.0f}, {:.0f}] | ".format(uav[0], uav[1], uav[2]), end="")
        print(
            "OFS[{:.0f},{:.0f}, {:.0f}] | ".format(offset[0], offset[1], offset[2]),
            end="",
        )
        print("ERR:{:.1f} | ".format(error), end="")
        print("TIME {:.2f} | ".format(elapsed))


if __name__ == "__main__":
    main()
