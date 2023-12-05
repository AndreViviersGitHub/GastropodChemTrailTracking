from collections import deque
import math
class Snail:
    def __init__(self, x=0, y=0, rotation=0):
        self.x = x
        self.y = y
        self.rotation = rotation  # Rotation in degrees
        self.initialRotation = rotation - 360
        self.positions = [(x, y)]  # Initially an empty list to store positions
        self.x_positions = deque(maxlen=5)  # Use 5 or another value for N
        self.y_positions = deque(maxlen=5)
        self.rotation_direction = ""

    def update_coordinates(self, x, y):
        # Update queues with new coordinates
        self.x_positions.append(x)
        self.y_positions.append(y)

        # Calculate average coordinates
        avg_x = sum(self.x_positions) / len(self.x_positions)
        avg_y = sum(self.y_positions) / len(self.y_positions)

        # Update the snail's position with the averaged values
        self.x = x
        self.y = y

    def update_rotation_and_position(self, rotation, maxRotation):
        rotation_difference = abs(rotation - self.rotation)  # Calculate the difference in rotation
        if rotation_difference > maxRotation and (self.x, self.y) not in self.positions:
            self.positions.append((self.x, self.y))
        self.rotation = rotation

    def update_rotation(self, rotation):
        self.rotation = rotation

    def get_position(self):
        return self.x, self.y

    def get_all_positions(self):
        return self.positions

    def get_rotation(self):
        return self.rotation

    def calculate_distance(self, new_x, new_y):
        """Calculate the distance between the current position and a new point."""
        return ((self.x - new_x) ** 2 + (self.y - new_y) ** 2) ** 0.5

    def significant_rotation_positions(self):
        """Return positions where the snail rotated more than 30 degrees from the previous position."""
        if not self.positions:
            return []

        significant_positions = [self.positions[0]]  # Always include the starting position
        last_point = self.positions[0]

        for current_point in self.positions[1:]:
            current_rotation = math.degrees(
                math.atan2(current_point[1] - last_point[1], current_point[0] - last_point[0]))
            last_rotation = math.degrees(
                math.atan2(last_point[1] - self.positions[0][1], last_point[0] - self.positions[0][0]))

            print(abs(angle_difference(current_rotation, last_rotation)))
            if abs(angle_difference(current_rotation, last_rotation)) >= 5:
                significant_positions.append(current_point)
                last_point = current_point  # update the last point to the current significant point

        return significant_positions

    def __str__(self):
        return f"Snail Current Position: ({self.x}, {self.y}), Rotation: {self.rotation}°"

def angle_difference(angle1, angle2):
    return (angle1 - angle2 + 180) % 360 - 180

# Example of usage
#snail = Snail(100, 100, 0)

#snail.update_coordinates(150, 150)
#snail.update_rotation(20)  # This won't add to positions list because rotation change is less than 30 degrees

#snail.update_coordinates(200, 200)
#snail.update_rotation(60)  # This will add to positions list because rotation change is more than 30 degrees

#(snail.get_all_positions())  # Prints the positions where rotation change was more than 30 degrees
