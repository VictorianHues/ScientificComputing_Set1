import numpy as np
import matplotlib.pyplot as plt

class InsertSink:
    def __init__(self, x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=False, **kwargs):
        """
        Add a sink into the domain, that has a given shape. 

        Args:
            x_length (float): The length of the domain along the x-axis. Default is 1.
            y_length (float): The length of the domain along the y-axis. Default is 1.
            n_steps (int): The number of steps (grid resolution) along both axes.
            shape_type (str): The type of shape to be implemented into the domain ('rectangle', 'triangle', or 'circle').
            show_plot (bool): If True, shows the plot with the added shape. Default is False.
            kwargs: Additional parameters specific to the shape.
        """
    
        self.x_length = x_length
        self.y_length = y_length
        self.n_steps = n_steps
        
        self.x = np.linspace(0, x_length, n_steps)
        self.y = np.linspace(0, y_length, n_steps)
        self.domain = np.ones((n_steps, n_steps))  
        
        self.shape_type = shape_type
        self.shape_params = kwargs
        self.show_plot = show_plot
        
        self.add_shape()

        if self.show_plot is True:
            self.show()

    def add_shape(self):
        """
        Adds the specified shape to the domain.
        """
        if self.shape_type == "rectangle":
            if "rectangles" in self.shape_params:
                self._add_rectangle(self.shape_params["rectangles"])
            else:
                raise ValueError("For 'rectangle', provide 'rectangles' as a list of (top_left, size).")
        elif self.shape_type == "triangle":
            if "triangle" in self.shape_params:
                self._add_triangle(self.shape_params["triangle"])
            else:
                raise ValueError("For 'triangle', provide 'triangle' as (top_vertex, height).")
        elif self.shape_type == "circle":
            if "circle" in self.shape_params:
                self._add_circle(self.shape_params["circle"])
            else:
                raise ValueError("For 'circle', provide 'circle' as (center, radius).")
        else:
            raise ValueError("Unsupported shape type. Choose 'rectangle', 'triangle', or 'circle'.")

    def _add_rectangle(self, rectangles):
        """
        Inserts multiple rectangles of zeros into the domain.

        Args:
            rectangles (list of tuples): Each tuple contains (top_left, size), 
                                         where top_left = (row, col) and size = (height, width).
        """
        for top_left, size in rectangles:
            row, col = top_left
            height, width = size
            
            row_end = min(row + height, self.domain.shape[0])
            col_end = min(col + width, self.domain.shape[1])

            self.domain[row:row_end, col:col_end] = 0

    def _add_triangle(self, triangle):
        """
        Function for adding a triangle into the domain.
        Args: 
            triangle: it contains (top_vertex, height), 
                top_vertex: coordinates of top vertex of the triangle.
                height: height of the triangle.
        """
        top_vertex, height = triangle
        row, col = top_vertex
        max_row = min(row + height, self.domain.shape[0])

        for i in range(height):
            start_col = max(col - i, 0)
            end_col = min(col + i + 1, self.domain.shape[1])
            if row + i < self.domain.shape[0]:
                self.domain[row + i, start_col:end_col] = 0

    def _add_circle(self, circle):
        """
        Function for adding a circle into the domain. 
        
        Args: 
            circle: it contains (center, radius), 
                center : coordinates of a center of the circle. 
                radius : radius of the circle 
        """
        center, radius = circle
        cx, cy = center
        rows, cols = self.domain.shape

        for x in range(max(0, cx - radius), min(rows, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(cols, cy + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.domain[x, y] = 0

    def show(self):
        """
        Function plots the domain with inserted shape, for visualization.
        """
        plt.imshow(self.domain, cmap="gray")
        plt.show()

# # Example 1: Multiple rectangles
# rectangles = [
#     ((10, 10), (10, 15)),  # First rectangle
#     ((20, 30), (10, 15))   # Second rectangle
# ]

# insert = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles=rectangles)

# # #Example 2: triangle

# triangle = ((25, 25), 10)
# insert = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="triangle", show_plot=True, triangle=triangle)

# # Example 3: circle
# circle = ((35, 15), 7)
# insert = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="circle", show_plot=True, circle=circle)