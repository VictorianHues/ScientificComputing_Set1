import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mask_domain import InsertSink
from SOR_diff import SORDiffusion
from main_masks import optimal_omega, main


# # EXPERIMENTS 

# #Experiment 1: Two rectangles - horizontal 

# domain_sink_two_rectangle_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((40,10), (5,10)), ((40,30),(5,10))])
# for_vis_two_v= InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,10), (5,10)), ((5,30),(5,10))])
# mask_rectangle_two_v = domain_sink_two_rectangle_v.domain
# optimal_omega_plot(mask = mask_rectangle_two_v)

# #Experiment 2: Two rectangles - vertical
# domain_sink_two_rectangle_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,15), (10,5)), ((35,30),(10,5))])
# for_vis_two_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,15), (10,5)), ((5,30),(10,5))])
# mask_rectangle_two_h = domain_sink_two_rectangle_h.domain
# optimal_omega_plot(mask = mask_rectangle_two_h) 

# #Experiment 3: Three rectangles - holizontal 

# domain_sink_three_rectangle_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((40,0), (5,10)), ((40,20),(5,10)), ((40,40),(5,10))])
# for_vis_three_h= InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,0), (5,10)), ((5,20), (5,10)), ((5,40), (5,10))])
# mask_rectangle_three_h = domain_sink_three_rectangle_h.domain
# optimal_omega_plot(mask = mask_rectangle_three_h)

# #Experiment 4: Three rectangles - vertical 

# domain_sink_three_rectangle_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,7), (10,5)), ((35,22),(10,5)), ((35,37),(10,5))])
# for_vis_three_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,7), (10,5)), ((5,22), (10,5)), ((5,37), (10,5))])
# mask_rectangle_three_v = domain_sink_three_rectangle_v.domain
# optimal_omega(mask = mask_rectangle_three_v)


# #Experiment 5: changing the size of the rectangle (only varing the x dimension)

# values = [(40,10),(35,15),(30,20),(25,25),(20,30),(15,35),(10,40),(5,45),(0,50)]

# results = {}

# for x_coordinate, size in values:
#     domain_sink_one_rectangle = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((x_coordinate,40), (size,10))])
#     mask_rectangle = domain_sink_one_rectangle.domain
#     w_min, N_min = optimal_omega_plot(mask = mask_rectangle)
#     results[(x_coordinate, size)] = (w_min, N_min)
#     print(results)

#Experiment 6: concentration profile - three - vertical
# tolerance = None

# domain_sink_three_rectangle_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,7), (10,5)), ((35,22),(10,5)), ((35,37),(10,5))])
# #for_vis_three_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,7), (10,5)), ((5,22), (10,5)), ((5,37), (10,5))])
# mask_rectangle_three_v = domain_sink_three_rectangle_v.domain
# #optimal_omega(mask = mask_rectangle_three_v)
# diff = SORDiffusion(x_length=1, y_length=1, n_steps=50,time_step_num = 20000, omega = 1.89, mask = mask_rectangle_three_v)
# c, end_time = diff.solve(tolerance)
# print(f"{end_time}")
# diff.plot_animation()
# diff.plot_single_frame(time = 10000)

#Experiment 7 - three -  horizontal

# tolerance = None
# domain_sink_three_rectangle_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((40,0), (5,10)), ((40,20),(5,10)), ((40,40),(5,10))])
# #for_vis_three_h= InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,0), (5,10)), ((5,20), (5,10)), ((5,40), (5,10))])
# mask_rectangle_three_h = domain_sink_three_rectangle_h.domain
# diff = SORDiffusion(x_length=1, y_length=1, n_steps=50,time_step_num = 20000, omega = 1.89, mask = mask_rectangle_three_h)
# c, end_time = diff.solve(tolerance)
# print(f"{end_time}")
# diff.plot_animation()
# diff.plot_single_frame(time = 10000)


# #Experiment 9- two - vertical
# domain_sink_two_rectangle_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,15), (10,5)), ((35,30),(10,5))])
# # for_vis_two_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,15), (10,5)), ((5,30),(10,5))])
# mask_rectangle_two_h = domain_sink_two_rectangle_h.domain
# tolerance = None
# diff = SORDiffusion(x_length=1, y_length=1, n_steps=50,time_step_num = 20000, omega = 1.89, mask = mask_rectangle_two_h)
# c, end_time = diff.solve(tolerance)
# diff.plot_animation()
# diff.plot_single_frame(time = 10000)


#Experiment 9- two - vertical
domain_sink_one_v = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((35,22), (10,5))])
# for_vis_two_h = InsertSink(x_length=1, y_length=1, n_steps=50, shape_type="rectangle", show_plot=True, rectangles = [((5,15), (10,5)), ((5,30),(10,5))])
mask_rectangle_two_h = domain_sink_one_v.domain
tolerance = None
diff = SORDiffusion(x_length=1, y_length=1, n_steps=50,time_step_num = 20000, omega = 1.89, mask = mask_rectangle_two_h)
c, end_time = diff.solve(tolerance)
diff.plot_animation()
diff.plot_single_frame(time = 10000)