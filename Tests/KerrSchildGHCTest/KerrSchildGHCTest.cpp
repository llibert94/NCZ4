/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

// General Chombo includes
// #include "BoxIterator.H"
#include "BoxLoops.hpp"
#include "Cell.hpp"
#include "ComputePack.hpp"
#include "DebuggingTools.hpp"
#include "FArrayBox.H"
#include "SetValue.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

// Problem specific for tests
//#include "AssignFixedBGtoBSSNVars.hpp"
//#include "CCZ4ScalarField.hpp"
#include "ExcisionTest.hpp"
#include "GammaCalculator.hpp"
#include "KerrSchild.hpp"
#include "GHCRHS.hpp"
//#include "MatterCCZ4.hpp"
//#include "MatterCCZ4RHS.hpp"
//#include "MatterEvolution.hpp"
//#include "NewConstraints.hpp"
//#include "ScalarField.hpp"
//#include "ScalarPotential.hpp"
#include "UserVariables.hpp"

int main()
{
#ifdef _OPENMP
    std::cout << "#threads = " << omp_get_max_threads() << std::endl;
#endif

    int failed = 0;
    const bool debug_plots_on = true; // false;
    const int num_resolutions = 1; // 0;

    // setup a vector of norms for checking convergence
    std::array<std::array<double, NUM_VARS>, num_resolutions> error_norms;  

    // loops over resolutions
    for (int ires = 0; ires < num_resolutions; ires++)
    {
        error_norms[ires].fill(0.0);
        // set up the array boxes for the vars inputs/outputs
        const int N_GRID = 128 * pow(2, ires);
        Box box(IntVect(0, 0, 0), IntVect(N_GRID - 1, N_GRID - 1, N_GRID - 1));
        Box ghosted_box(IntVect(-3, -3, -3),
                        IntVect(N_GRID + 2, N_GRID + 2, N_GRID + 2));
        Box doubleghosted_box(IntVect(-6, -6, -6),
                              IntVect(N_GRID + 5, N_GRID + 5, N_GRID + 5));
        FArrayBox fixedbg_fab(doubleghosted_box, NUM_VARS);
        BoxLoops::loop(make_compute_pack(SetValue(0.0)), fixedbg_fab,
                       fixedbg_fab);
        FArrayBox deriv_fixedbg_fab(ghosted_box, NUM_VARS);
        BoxLoops::loop(make_compute_pack(SetValue(0.0)), deriv_fixedbg_fab,
                       deriv_fixedbg_fab);
        FArrayBox rhs_fab(box, NUM_VARS);
        BoxLoops::loop(make_compute_pack(SetValue(0.0)), rhs_fab, rhs_fab);
        FArrayBox fixedbg_rhs_fab(box, NUM_VARS);
        BoxLoops::loop(make_compute_pack(SetValue(0.0)), fixedbg_rhs_fab,
                       fixedbg_rhs_fab);

        // grid properties
        const double length = 16.0;
        const double dx = length / (N_GRID);
        const double center = length / 2.0;
        const std::array<double, CH_SPACEDIM> center_vector = {center, center,
                                                               center};

        // Test the fixed BG - first assign the fixed bg vars to the BSSN vars
        KerrSchild::params_t bg_params;
        bg_params.mass = 2.0;
        bg_params.spin = 0.5;
        bg_params.center = center_vector;
        KerrSchild kerr_bh(bg_params, dx);
        BoxLoops::loop(
            KerrSchild(bg_params, dx),
            fixedbg_fab, fixedbg_fab);
        // used temp single ghosted box to avoid nans at boundaries in Gamma^i
        BoxLoops::loop(GammaCalculator(dx, center_vector, 1), fixedbg_fab, deriv_fixedbg_fab);
        fixedbg_fab += deriv_fixedbg_fab;

        // Get the Ham and Mom constraints using these values and finite diffs
        // Put them in the rhs (although they aren't rhs)
        //BoxLoops::loop(Constraints(dx, c_Ham, Interval(c_Mom1, c_Mom3)),
        //               fixedbg_fab, rhs_fab, box);

        // Calculate the RHS using finite differences for the derivs
        const double G_Newton = 0.0; // ignore backreaction
        const double sigma = 0.0;    // no kreiss oliger
        GHCRHS<>::params_t ghc_params;
        ghc_params.kappa1 = 0.0;
        ghc_params.kappa2 = 0.0;
        //ghc_params.kappa3 = 0.0;
        ghc_params.lapse_coeff = 0.0; // no evolution lapse or shift
        ghc_params.shift_Gamma_coeff = 0.0;
	ghc_params.eta = 0.0;

        //const double scalar_mass = 0.1;
        //ScalarPotential potential(scalar_mass);
        //CCZ4ScalarField<ScalarPotential> scalar_field(potential);

        /*BoxLoops::loop(MatterCCZ4RHS<CCZ4ScalarField<ScalarPotential>>(
                           scalar_field, ccz4_params, dx, sigma,
                           CCZ4RHS<>::USE_BSSN, G_Newton),
                       fixedbg_fab, rhs_fab);
	*/
        // Calculate the Matter RHS using the analytic derivatives
        //ScalarField<ScalarPotential> fixed_scalar_field(potential);
        //MatterEvolution<ScalarField<ScalarPotential>, KerrSchild> my_evolution(
        //    fixed_scalar_field, kerr_bh, sigma, dx, center_vector);
	GHCRHS<> my_evolution(ghc_params, dx, sigma, center_vector, 1);
        BoxLoops::loop(make_compute_pack(my_evolution), fixedbg_fab,
                       fixedbg_rhs_fab);

        // take the difference for the Matter RHS, which should converge to zero
        rhs_fab -= fixedbg_rhs_fab;

        // Excise the centre within the horizon where there are always large
        // values
        BoxLoops::loop(ExcisionTest<KerrSchild>(
                           dx, center_vector, kerr_bh),
                       rhs_fab, rhs_fab, disable_simd());
	
        // Output slice of data on lowest res, useful for debugging
        // activate by setting debug_plots to true above
        if (ires == 0 && debug_plots_on)
        {
            //std::string filename = "output.txt";
            //std::ofstream outfile;
            //outfile.clear();
            //outfile.open(filename);
            //outfile << std::setw(20) << "# x , z , value"
            //        << "\n";
            BoxIterator bit(box);
            for (bit.begin(); bit.ok(); ++bit)
            {
                // work out location on the grid, taking slice through center
                IntVect iv = bit();
                if (iv[0] == N_GRID / 4 + N_GRID / 2 && iv[1] == N_GRID / 4 + N_GRID / 2 && iv[2] == 0 + N_GRID / 2)
                {
                    double x = dx * (iv[0] + 0.5) - center_vector[0];
		    double y = dx * (iv[1] + 0.5) - center_vector[1]; 
                    double z = dx * (iv[2] + 0.5) - center_vector[2];
                    double out1 = fixedbg_fab(iv, c_Gam1);
		    double out2 = fixedbg_rhs_fab(iv, c_g11);
                    double out3 = fixedbg_rhs_fab(iv, c_Pi);

                    std::cout << x << ' ' << y << ' ' << z << std::endl;
                    std::cout << out1 << ' ' << out2 << ' ' << out3 << std::endl;
                    //outfile << "\n";
                }
            }
            //outfile.close();
        }

        // Checking the results
        const int max_norm = 0;
        const int L1_norm = 1;
        const int num_comps = 1;
        const double error_limit = 0.001;
        // check that you have zero Ham and Mom with the initial data
        // such that it satisfies the constraints
        /*for (int i = c_Ham; i <= c_Mom3; ++i)
        {
            // first check for large non zero values outside horizon
            double max_err = rhs_fab.norm(max_norm, i, num_comps);
            if (max_err > error_limit)
            {
                std::cout << "CONSTRAINT " << UserVariables::variable_names[i]
                          << " IS NON ZERO: MAX ERROR = " << max_err
                          << std::endl;
                failed = -1;
            }
            // save the L1 norm for the convergence check - gives sum of abs
            // values
            error_norms[ires][i] =
                rhs_fab.norm(L1_norm, i, num_comps) * pow(N_GRID, -3);
        }*/

        // compare the rhs for the scalar field using the calculated derivs
        // versus the finite difference case - this tests the expressions
        // for d1_lapse and d1_gamma etc
 /*       for (int i = c_g11; i <= c_B3; ++i)
        {
            // first check for large non zero values outside horizon
            double max_err = rhs_fab.norm(max_norm, i, num_comps);
            if (max_err > error_limit)
            {
                std::cout
                    << "ANALYTIC MATTER VARS RHS FOR "
                    << UserVariables::variable_names[i]
                    << " DOES NOT MATCH FINITE DIFFERENCE RHS: MAX ERROR = "
                    << max_err << std::endl;
                failed = -1;
            }
            // save the L1 norm for the convergence check - gives sum of abs
            // values
            error_norms[ires][i] =
                rhs_fab.norm(L1_norm, i, num_comps) * pow(N_GRID, -3);
        }
*/
        // check that the RHS for the metric vars is zero in this gauge
        // otherwise you do not have a valid static BG...
        
	for (int i = c_g11; i <= c_B3; ++i)
        {
            // first check for large non zero values outside horizon
            double max_err = rhs_fab.norm(max_norm, i, num_comps);
            if (max_err > error_limit)
            {
                std::cout << "RHS FOR COMPONENT "
                          << UserVariables::variable_names[i]
                          << " IS NON ZERO: MAX ERROR = " << max_err
                          << std::endl;
                failed = -1;
            }
	    /*else
	    {
		std::cout << "RHS FOR COMPONENT "
                          << UserVariables::variable_names[i]
                          << " IS ZERO: MAX ERROR = " << max_err
                          << std::endl;
	    }*/
            // save the L1 norm for the convergence check - gives sum of abs
            // values
            error_norms[ires][i] =
                rhs_fab.norm(L1_norm, i, num_comps) * pow(N_GRID, -3);
        }
    }

    // Check convergence to zero with increasing resolution
    /*double min_convergence_factor = 16.0;
    for (int i = 0; i < NUM_VARS; ++i)
    {
        for (int ires = 0; ires < num_resolutions - 1; ires++)
        {
            double hi_res_norm = error_norms[ires + 1][i];
            double lo_res_norm = error_norms[ires][i];
            // ignore the exact zero values
            if (abs(hi_res_norm) < 1e-16 && abs(lo_res_norm) < 1e-16)
            {
                lo_res_norm = 1e-8;
                hi_res_norm = 1e-10;
            }
            double convergence_factor = lo_res_norm / hi_res_norm;

            // demand at least 3.5 order convergence
            // (should be nearly 16, in general is around 13-15)
            if (convergence_factor < min_convergence_factor)
            {
                min_convergence_factor = convergence_factor;
            }
            if (convergence_factor < 11)
            {
                failed = -1;
                std::cout << "CONVERGENCE FACTOR FOR COMPONENT "
                          << UserVariables::variable_names[i] << " ON LEVEL "
                          << ires << " IS LOW: VALUE = " << convergence_factor
                          << " " << hi_res_norm << " " << lo_res_norm
                          << std::endl;
            }
        }
    }

    // Check failure, can be from very large errors, or from failure to converge
    if (failed == 0)
    {
        std::cout << "The minimum convergence factor was "
                  << min_convergence_factor << std::endl;
        std::cout << "Fixed Background test passed..." << std::endl;
    }
    else
    {
        std::cout << "The minimum convergence factor was "
                  << min_convergence_factor << std::endl;
        std::cout << "Fixed Background test failed..." << std::endl;
    }*/
    return failed;
}
