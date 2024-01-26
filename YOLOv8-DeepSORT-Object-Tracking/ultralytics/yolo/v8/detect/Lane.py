class Lane:

    def __init__(self, rect, max_x, max_y, enter, out):
        self.rect = rect
        self.max_x = max_x
        self.max_y = max_y
        self.enter = enter
        self.out = out
        self.cars = []

    def is_point_inside(self, point):
        x, y = point
        xx1, yy1, w, h = self.rect
        xx2, yy2 = xx1 + w, yy1 + h

        # Assuming max_x and max_y are the maximum x and y coordinates in your image
        return xx1 <= x <= xx2 and yy1 <= y <= yy2

    def get_start_point(self):
        return self.coordinates[0], self.coordinates[1]

    def get_end_point(self):
        return self.coordinates[0] + self.coordinates[2], self.coordinates[1] + self.coordinates[3]

    def get_coordinates(self):
        return self.coordinates

    def get_origin(self):
        return self.x_origin, self.y_origin

    def contains_point(self, point):
        x, y = point
        start_x, start_y = self.get_start_point()
        end_x, end_y = self.get_end_point()

        return start_x <= x <= end_x and start_y <= y <= end_y
