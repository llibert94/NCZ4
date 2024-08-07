/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GHCVARS_HPP_
#define GHCVARS_HPP_

#include "Tensor.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"

/// Namespace for GHC vars
/** The structs in this namespace collect all the GHC variables. It's main use
 *  is to make a local, nicely laid-out, copy of the GHC variables for the
 *  current grid cell (Otherwise, this data would only exist on the grid in
 *  the huge, flattened Chombo array).
 **/
namespace GHCVars
{
/// Vars object for GHC vars, including gauge vars
template <class data_t> struct VarsNoGauge
{
    data_t Theta;          //!< GHC quantity associated to Hamiltonian constraint
    Tensor<2, data_t> g;   //!< Physical metric
    Tensor<2, data_t> K;   //!< Extrinsic curvature
    Tensor<1, data_t> Gam; //!< Physical Gamma^i variable

    /// Defines the mapping between members of Vars and Chombo grid
    /// variables (enum in User_Variables)
    template <typename mapping_function_t>
    void enum_mapping(mapping_function_t mapping_function)
    {
        using namespace VarsTools; // define_enum_mapping is part of VarsTools
	define_enum_mapping(mapping_function, c_Theta, Theta);
	// Symmetric 2-tensors
        define_symmetric_enum_mapping(
	   mapping_function, GRInterval<c_g11, D_SELECT(, c_g22, c_g33)>(), g);
	define_symmetric_enum_mapping(
           mapping_function, GRInterval<c_K11, D_SELECT(, c_K22, c_K33)>(), K);
	define_enum_mapping(
            mapping_function, GRInterval<c_Gam1, D_SELECT(, c_Gam2, c_Gam3)>(),
            Gam); //!< The auxilliary variable Gamma^i

    }
};

/// Vars object for GHC vars, including gauge vars
template <class data_t>
struct VarsWithGauge : public VarsNoGauge<data_t>
{
    data_t lapse;
    Tensor<1, data_t> shift;
    Tensor<1, data_t> B;

    /// Defines the mapping between members of Vars and Chombo grid
    /// variables (enum in User_Variables)
    template <typename mapping_function_t>
    void enum_mapping(mapping_function_t mapping_function)
    {
        using namespace VarsTools; // define_enum_mapping is part of VarsTools
        VarsNoGauge<data_t>::enum_mapping(mapping_function);
        define_enum_mapping(mapping_function, c_lapse, lapse);
        define_enum_mapping(
            mapping_function,
            GRInterval<c_shift1, D_SELECT(, c_shift2, c_shift3)>(), shift);
        define_enum_mapping(mapping_function,
                            GRInterval<c_B1, D_SELECT(, c_B2, c_B3)>(), B);
    }
};

/// Vars object for GHC vars needing second derivs, excluding gauge vars
template <class data_t>
struct Diff2VarsNoGauge
{
    Tensor<2, data_t> g; //!< Physical metric

    template <typename mapping_function_t>
    void enum_mapping(mapping_function_t mapping_function)
    {
        using namespace VarsTools; // define_enum_mapping is part of VarsTools
        define_symmetric_enum_mapping(
            mapping_function, GRInterval<c_g11, D_SELECT(, c_g22, c_g33)>(), g);
    }
};

/// Vars object for GHC vars needing second derivs, including gauge vars
template <class data_t>
struct Diff2VarsWithGauge : public Diff2VarsNoGauge<data_t>
{
    data_t lapse;
    Tensor<1, data_t> shift;

    /// Defines the mapping between members of Vars and Chombo grid
    /// variables (enum in User_Variables)
    template <typename mapping_function_t>
    void enum_mapping(mapping_function_t mapping_function)
    {
        using namespace VarsTools; // define_enum_mapping is part of VarsTools
        Diff2VarsNoGauge<data_t>::enum_mapping(mapping_function);
        define_enum_mapping(mapping_function, c_lapse, lapse);
        define_enum_mapping(
            mapping_function,
            GRInterval<c_shift1, D_SELECT(, c_shift2, c_shift3)>(), shift);
    }
};
} // namespace GHCVars

#endif /* GHCVARS_HPP_ */
