import numpy as np
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

domain_sink_rectangle_2 = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", rectangles = [((10,10), (10,15)), ((20,25),(10,15))])
mask = domain_sink_rectangle_2.domain

optimal_omega_plot(mask = mask, title='Two rectangles mask', file ='plots/opt_omega_full_rectangle_2_1.png')

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