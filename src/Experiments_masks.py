import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mask_domain import InsertSink
from SOR_diff import SORDiffusion
from main_masks import optimal_omega_plot, main

# triangle 

# domain_sink_triangle = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="triangle", triangle = ((10,10), 5))
# mask = domain_sink_triangle.domain

# optimal_omega_plot(mask = mask, title='Triangle mask', file ='plots/opt_omega_full_triangle.png')

#rectangle 

#domain_sink_rectangle = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", rectangles = [((10,10), (10,15))])
#mask = domain_sink_rectangle.domain

#optimal_omega_plot(mask = mask, title='Rectangle mask', file ='plots/opt_omega_full_rectangle.png')

#animation with mask - check
# mask = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles= [((35,30), (10,15))])
# domain = mask.domain
# main(domain)

#Two rectangles

# domain_sink_rectangle_2 = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", rectangles = [((10,10), (10,15)), ((20,25),(10,15))])
# mask = domain_sink_rectangle_2.domain

# optimal_omega_plot(mask = mask, title='Two rectangles mask', file ='plots/opt_omega_full_rectangle_2_1.png')

#Three rectangles

# domain_sink_rectangle_3 = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", rectangles = [((10,10), (10,15)), ((20,25),(10,15)), ((25,5),(10,15))])
# mask = domain_sink_rectangle_3.domain

# optimal_omega_plot(mask = mask, title='Three rectangles mask', file ='plots/opt_omega_full_rectangle_3.png')

## omega depending on the size of the object 

# size_array = [(i,i) for i in range(3,40)]

# for size in size_array: 
#     domain_sink_rectangle = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", rectangle = ((3,3), size))
#     mask = domain_sink_rectangle.domain
#     optimal_omega_plot(mask = mask, title='Rectangle mask', file ='plots/opt_omega_full_rectangle.png')


#PROPER EXPERIMENTS 

#Experiments 0: 

# optimal_omega_plot(mask = None)

#Experiment 1: One rectangles - recreate!!!

# domain_sink_one_rectangle = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((30,30), (10,15))])
# mask_rectangle = domain_sink_one_rectangle.domain
# optimal_omega_plot(mask = mask_rectangle)

#Experiment 2: Two rectangles - horizontal 

# domain_sink_two_rectangle_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((40,10), (5,10)), ((40,30),(5,10))])
# for_vis_two_v= InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,10), (5,10)), ((5,30),(5,10))])
# mask_rectangle_two_v = domain_sink_two_rectangle_v.domain
# optimal_omega_plot(mask = mask_rectangle_two_v)

#Experiment 3: Two rectangles - vertical
# domain_sink_two_rectangle_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,15), (10,5)), ((35,30),(10,5))])
# for_vis_two_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,15), (10,5)), ((5,30),(10,5))])
# mask_rectangle_two_h = domain_sink_two_rectangle_h.domain
# optimal_omega_plot(mask = mask_rectangle_two_h) 

#Experiment 4: Three rectangles - holizontal 

# domain_sink_three_rectangle_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((40,0), (5,10)), ((40,20),(5,10)), ((40,40),(5,10))])
# for_vis_three_h= InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,0), (5,10)), ((5,20), (5,10)), ((5,40), (5,10))])
# mask_rectangle_three_h = domain_sink_three_rectangle_h.domain
# optimal_omega_plot(mask = mask_rectangle_three_h)

#Experiment 5: Three rectangles - vertical 

# domain_sink_three_rectangle_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,7), (10,5)), ((35,22),(10,5)), ((35,37),(10,5))])
# for_vis_three_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,7), (10,5)), ((5,22), (10,5)), ((5,37), (10,5))])
# mask_rectangle_three_v = domain_sink_three_rectangle_v.domain
# optimal_omega_plot(mask = mask_rectangle_three_v)

#Experiment 6: changing the size of the rectangle (only varing the x dimension)

# values = [(40,10),(35,15),(30,20),(25,25),(20,30),(15,35),(10,40),(5,45),(0,50)]

# results = {}

# for x_coordinate, size in values:
#     domain_sink_one_rectangle = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((x_coordinate,40), (size,10))])
#     mask_rectangle = domain_sink_one_rectangle.domain
#     w_min, N_min = optimal_omega_plot(mask = mask_rectangle)
#     results[(x_coordinate, size)] = (w_min, N_min)
#     print(results)

#Plot: 

def plot_multiple_rectangles(values, n_steps=50):
    """
    Plots multiple rectangles with blue boundaries and light blue interiors, 
    with the top and bottom reversed.
    
    Args:
        values (list of tuples): Each tuple contains (x_coordinate, size).
        n_steps (int): Grid resolution along both axes.
    """
    plt.figure(figsize=(6, 6))

    # For each rectangle, plot its boundary and fill the interior
    for x_coordinate, size in values:
        sink = InsertSink(x_length=1, y_length=1, n_steps=n_steps, shape_type="rectangle", 
                          rectangles=[((x_coordinate, 40), (size, 10))])
        
        # Get rectangle parameters
        for top_left, size in [((x_coordinate, 40), (size, 10))]:
            row, col = top_left
            height, width = size

            # Flip the row to reverse the top and bottom
            flipped_row = n_steps - (row + height)

            # Plot the boundary as lines (rectangle edges)
            plt.plot([col, col + width], [flipped_row, flipped_row], color='blue', lw=2)  # top edge
            plt.plot([col, col + width], [flipped_row + height, flipped_row + height], color='blue', lw=2)  # bottom edge
            plt.plot([col, col], [flipped_row, flipped_row + height], color='blue', lw=2)  # left edge
            plt.plot([col + width, col + width], [flipped_row, flipped_row + height], color='blue', lw=2)  # right edge

            # Fill the interior with light blue (reversed)
            plt.fill_between([col, col + width], flipped_row, flipped_row + height, color='lightblue', zorder=-1)

    # Set plot limits and labels
    plt.xlim(0, n_steps)
    plt.ylim(0, n_steps)
    plt.gca().invert_yaxis()  # Ensure the y-axis is inverted, so (0,0) is at the top-left corner
    plt.grid()
    plt.show()

# Define rectangle values
values = [(40,10), (35,15), (30,20), (25,25), (20,30), (15,35), (10,40), (5,45), (0,50)]

# Call the function to visualize all rectangle boundaries and filled interiors with top and bottom reversed
plot_multiple_rectangles(values)

