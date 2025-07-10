class Boundaries:
    def __init__(self, total_size, fov_size):

        self.start_middle_x = round(total_size[0] / 2) - round(fov_size[0] / 2)
        self.end_middle_x = self.start_middle_x + fov_size[0]

        self.start_middle_y = round(total_size[1] / 2) - round(fov_size[1] / 2)
        self.end_middle_y = self.start_middle_y + fov_size[1]

        self.end_left_y = round(total_size[1] / 2) - round(fov_size[1] / 2)
        self.start_left_y = self.end_left_y - fov_size[1]

        self.start_right_y = round(total_size[1] / 2) + round(fov_size[1] / 2)
        self.end_right_y = self.start_right_y + fov_size[1]

        self.start_bottom_x = round(total_size[0] / 2) + round(fov_size[0] / 2)
        self.end_bottom_x = self.start_bottom_x + fov_size[0]

        self.end_top_x = round(total_size[0] / 2) - round(fov_size[0] / 2)
        self.start_top_x = self.end_top_x - fov_size[0]

        if self.start_left_y < 0:
            self.start_left_y = 0

        if self.end_right_y > total_size[1]:
            self.end_right_y = total_size[1]

        if self.start_top_x < 0:
            self.start_top_x = 0

        if self.end_bottom_x > total_size[0]:
            self.end_bottom_x = total_size[0]
