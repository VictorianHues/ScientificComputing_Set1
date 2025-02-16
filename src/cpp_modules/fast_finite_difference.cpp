/*
    Course: Complex systems
    Names: Marvin Frommer, Wessel Beumer, Paul Jungnickel, Tika van Bennekum

    File description:
        This file contains the system in C++.
*/

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include<omp.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* 
compute sum of an nparray for testing if the bindings work 
*/
double sum(py::array_t<double> &a)
{

    auto a_arr = a.unchecked<2>();

    double suma{0.0};

    for (py::ssize_t i = 0; i < a_arr.shape(0); ++i)
    {
        for (py::ssize_t j = 0; j < a_arr.shape(1); ++j)
        {
            suma += a_arr(i, j);
        }
    }

    return suma;
}

/*
convert a python dict to a cpp map<string, double>
*/
std::map<std::string, double> convert_dict(py::dict dict)
{
    std::map<std::string, double> converted_map;
    for (auto elem : dict)
    {
        std::string first = std::string(py::str(elem.first));
        double second = elem.second.cast<double>();
        converted_map.insert(std::pair<std::string, double>(first, second));
    }
    return converted_map;
}



/*
Twin of jacobi in finite_difference.py

Params:
    c_old, c_new: [height x width] ndarray
    max_iter: number of iterations
    width: size of arrays
    tolerance: stopping condition
*/
int fast_jacobi(py::array_t<double> & c_old_np, py::array_t<double> & c_new_np, size_t max_iter, size_t width, double tolerance){

    auto c_old = c_old_np.mutable_unchecked<2>();
    auto c_new = c_new_np.mutable_unchecked<2>();

    for(py::size_t t = 0; t<max_iter; t++){

        for(py::size_t j = 0; j<width; j++){
            c_old(j, width-1) = 1;
            c_old(j, 0) = 0;
        }
        double delta = 0;
        #pragma omp parallel for
        for(py::size_t i=0; i<width; i++){
            for(py::size_t j=1; j<width-1; j++){
                
                c_new(i,j) = 0.25* (                    
                    c_old((i+1) % width, j) +
                    c_old((i-1) % width, j) +
                    c_old(i, j+1) +
                    c_old(i, j-1)
                );
                delta = std::max(delta, std::abs(c_new(i,j) - c_old(i,j)));

            }
        }

        // std::cout << delta << std::endl;
        if (delta < tolerance){
            return t;
        }
        // could not get swapping to work
        #pragma omp parallel for
        for(py::size_t i=0; i<width; i++){
            for(py::size_t j=0; j<width; j++){                
                c_old(i,j)  = c_new(i,j);
            }
        }

    }
    return max_iter;
}



/*
Simulate the Cellular Automata model. This method was written with the same functionality as CA.py run_simulation.
The simulation states are saved in the grids array which is passed by reference.

Params:
    grids: [num_steps x height x width x 2] ndarray
            first entry must be initial state grid
            first / last row of initial state is the boundary condition
    params: pyton dict with parameters (must contain entries for EROSION_(K|C|m|n))
*/
void simulate(py::array_t<double> &grids, py::dict params)
{

    auto param_map = convert_dict(params);

    const auto EROSION_K = param_map["EROSION_K"];
    const auto EROSION_C = param_map["EROSION_C"];
    const auto EROSION_n = param_map["EROSION_n"];
    const auto EROSION_m = param_map["EROSION_m"];

    auto sgn = [&](double x) -> double
    {
        return (x > 0) - (x < 0);
    };

    auto erosion_rule = [&](double Q, double S) -> double
    {
        if (Q <= 0)
        {
            return 0;
        }
        auto QSC = Q * (abs(S) + EROSION_C);
        return EROSION_K * sgn(S) * std::min(EROSION_C, std::pow(QSC, EROSION_m));
    };

    auto grid_arr = grids.mutable_unchecked<4>();

    auto num_steps = grid_arr.shape(0);
    auto height = grid_arr.shape(1);
    auto width = grid_arr.shape(2);
    auto NUM_CELL_FLOATS = grid_arr.shape(3);
    ssize_t GROUND_HEIGHT = 0;
    ssize_t WATER_HEIGHT = 1;

    auto enforce_boundary = [&](ssize_t curr_step)
    {
        for (ssize_t col = 0; col < width; col++)
        {
            grid_arr(curr_step, 0, col, WATER_HEIGHT) = grid_arr(0, 0, col, WATER_HEIGHT);
            grid_arr(curr_step, 0, col, GROUND_HEIGHT) = grid_arr(0, 0, col, GROUND_HEIGHT);
            grid_arr(curr_step, height - 1, col, WATER_HEIGHT) = grid_arr(0, height - 1, col, WATER_HEIGHT);
            grid_arr(curr_step, height - 1, col, GROUND_HEIGHT) = grid_arr(0, height - 1, col, GROUND_HEIGHT);
        }
        return;
    };

    auto copy_state = [&](ssize_t to_step, ssize_t from_step)
    {
        if (to_step != from_step)
        {
            for (ssize_t row = 0; row < height; row++)
            {
                for (ssize_t col = 0; col < width; col++)
                {
                    grid_arr(to_step, row, col, WATER_HEIGHT) = grid_arr(from_step, row, col, WATER_HEIGHT);
                    grid_arr(to_step, row, col, GROUND_HEIGHT) = grid_arr(from_step, row, col, GROUND_HEIGHT);
                }
            }
        }
        return;
    };

    auto apply_rule = [&](ssize_t curr_step, ssize_t prev_step, ssize_t row, ssize_t col)
    {
        if (grid_arr(curr_step, row, col, WATER_HEIGHT) <= 0)
        {
            return;
        }

        std::vector<std::tuple<double, ssize_t, ssize_t>> pos_slopes;
        std::vector<std::tuple<double, ssize_t, ssize_t>> zero_slopes;
        std::vector<std::tuple<double, ssize_t, ssize_t>> neg_slopes;

        auto add_to_list = [&](double slope, ssize_t nrow, ssize_t ncol)
        {
            auto tuple = std::make_tuple(slope, nrow, ncol);
            if (slope > 0)
            {
                pos_slopes.push_back(tuple);
            }
            else if (slope < 0)
            {
                neg_slopes.push_back(tuple);
            }
            else
            {
                zero_slopes.push_back(tuple);
            }
        };

        if (row < height - 1)
        {
            // bot left
            if (col > 0)
            {
                double slope = M_SQRT1_2 * (grid_arr(prev_step, row, col, GROUND_HEIGHT) - grid_arr(prev_step, row + 1, col - 1, GROUND_HEIGHT));
                add_to_list(slope, row + 1, col - 1);
            }
            // bottom neighbor
            double slope = M_SQRT1_2 * (grid_arr(prev_step, row, col, GROUND_HEIGHT) - grid_arr(prev_step, row + 1, col, GROUND_HEIGHT));
            add_to_list(slope, row + 1, col);
            // bot right
            if (col < width - 1)
            {
                double slope = M_SQRT1_2 * (grid_arr(prev_step, row, col, GROUND_HEIGHT) - grid_arr(prev_step, row + 1, col + 1, GROUND_HEIGHT));
                add_to_list(slope, row + 1, col + 1);
            }
        }

        if (pos_slopes.size() > 0)
        {
            double slope_sum = 0;
            for (auto elem : pos_slopes)
            {
                slope_sum += std::pow(std::get<0>(elem), EROSION_n);
            }
            for (auto elem : pos_slopes)
            {
                auto [slope, nrow, ncol] = elem;
                auto proportion = std::pow(slope, EROSION_n) / slope_sum;
                auto discharge = grid_arr(prev_step, row, col, WATER_HEIGHT) * proportion;
                auto erosion = erosion_rule(discharge, slope);
                grid_arr(curr_step, row, col, WATER_HEIGHT) -= discharge;
                grid_arr(curr_step, nrow, ncol, WATER_HEIGHT) += discharge;
                grid_arr(curr_step, row, col, GROUND_HEIGHT) -= erosion;
                grid_arr(curr_step, nrow, ncol, GROUND_HEIGHT) += erosion;
            }
        }
        else if (zero_slopes.size() > 0)
        {
            for (auto elem : zero_slopes)
            {
                auto [slope, nrow, ncol] = elem;
                auto proportion = slope / zero_slopes.size();
                auto discharge = grid_arr(prev_step, row, col, WATER_HEIGHT) * proportion;
                auto erosion = erosion_rule(discharge, slope);
                grid_arr(curr_step, row, col, WATER_HEIGHT) -= discharge;
                grid_arr(curr_step, nrow, ncol, WATER_HEIGHT) += discharge;
                grid_arr(curr_step, row, col, GROUND_HEIGHT) -= erosion;
                grid_arr(curr_step, nrow, ncol, GROUND_HEIGHT) += erosion;
            }
        }
        else
        {
            double slope_sum = 0;
            for (auto elem : neg_slopes)
            {
                slope_sum += std::pow(std::abs(std::get<0>(elem)), -EROSION_n);
            }
            for (auto elem : neg_slopes)
            {
                auto [slope, nrow, ncol] = elem;
                auto proportion = std::pow(std::abs(slope), -EROSION_n) / slope_sum;
                auto discharge = grid_arr(prev_step, row, col, WATER_HEIGHT) * proportion;
                auto erosion = erosion_rule(discharge, slope);
                grid_arr(curr_step, row, col, WATER_HEIGHT) -= discharge;
                grid_arr(curr_step, nrow, ncol, WATER_HEIGHT) += discharge;
                grid_arr(curr_step, row, col, GROUND_HEIGHT) -= erosion;
                grid_arr(curr_step, nrow, ncol, GROUND_HEIGHT) += erosion;
            }
        }
    };

    for (ssize_t step = 1; step < num_steps; step++)
    {
        auto prev_step = step - 1;
        copy_state(step, prev_step);

        for (ssize_t row = 0; row < height - 1; row++)
        {
            for (ssize_t col = 0; col < width; col++)
            {
                apply_rule(step, prev_step, row, col);
            }
        }
        enforce_boundary(step);

    }

    return;
}

PYBIND11_MODULE(fast_finite_difference, m)
{
    m.doc() = "sum the elements of a 2-dimensional array";
    m.def("sum", &sum);
    m.def("simulate", &simulate);
    m.def("fast_jacobi", &fast_jacobi);
}
