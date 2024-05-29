/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GHCVARIABLES_HPP
#define GHCVARIABLES_HPP

#include <algorithm>
#include <array>
#include <string>

/// This enum gives the index of the CCZ4 variables on the grid
enum
{
    c_g11,
    c_g12,
    c_g13,
    c_g22,
    c_g23,
    c_g33,

    c_K11,
    c_K12,
    c_K13,
    c_K22,
    c_K23,
    c_K33,

    c_Theta,

    c_Gam1,
    c_Gam2,
    c_Gam3,

    c_lapse,

    c_shift1,
    c_shift2,
    c_shift3,

    c_B1,
    c_B2,
    c_B3,

    NUM_GHC_VARS
};

namespace UserVariables
{
static const std::array<std::string, NUM_GHC_VARS> ghc_variable_names = {
    "g11",    "g12",    "g13",    "g22", "g23", "g33",

    "K11",    "K12",    "K13",    "K22", "K23", "K33",

    "Theta",

    "Gam1", "Gam2", "Gam3",

    "lapse",

    "shift1", "shift2", "shift3",

    "B1",     "B2",     "B3",
};
} // namespace UserVariables

#endif /* GHCVARIABLES_HPP */
