/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

// General includes
#include "GRParmParse.hpp"
#include "SimulationParametersBase.hpp"

// Problem specific includes:
#include "KerrSchildAdS.hpp"
//#include "Minkowski.hpp"
#include "PoincareAdS.hpp"

class SimulationParameters : public SimulationParametersBase<PoincareAdS>
{
  public:
    SimulationParameters(GRParmParse &pp) : SimulationParametersBase(pp)
    {
        read_params(pp);
        check_params();
    }

    /// Read parameters from the parameter file
    void read_params(GRParmParse &pp)
    {
        // Initial Kerr data
        pp.load("black_brane_length", black_brane_params.length);
        pp.load("black_brane_radius", black_brane_params.z0);
        pp.load("black_brane_center", black_brane_params.center, center);

        pp.load("activate_extraction", activate_extraction, false);

        // Background
        pp.load("bg_length", bg_params.length);
        pp.load("bg_center", bg_params.center, center);

#ifdef USE_AHFINDER
        pp.load("AH_initial_guess", AH_initial_guess, black_brane_params.z0);
#endif
    }

    void check_params()
    {
        warn_parameter("black_brane_length", black_brane_params.length, black_brane_params.length >= 0.0,
                       "should be >= 0.0");
        check_parameter("black_brane_radius", black_brane_params.z0,
                        black_brane_params.z0 <= black_brane_params.length,
                        "must satisfy z0 <= L");
        FOR(idir)
        {
            std::string name = "black_brane_center[" + std::to_string(idir) + "]";
            warn_parameter(
                name, black_brane_params.center[idir],
                (black_brane_params.center[idir] >= 0) &&
                    (black_brane_params.center[idir] <= (ivN[idir] + 1) * coarsest_dx),
                "should be within the computational domain");
        }
    }

    PoincareAdS::params_t bg_params;
    KerrSchildAdS<PoincareAdS>::params_t black_brane_params;
    bool activate_extraction;
#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
