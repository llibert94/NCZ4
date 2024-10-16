
/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef KERRSCHILD_HPP_
#define KERRSCHILD_HPP_

#include "GHCVars.hpp"
#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "simd.hpp"

//! Class which computes the initial conditions for a Kerr Schild BH
//! https://arxiv.org/pdf/gr-qc/9805023.pdf
//! https://arxiv.org/pdf/2011.07870.pdf

class KerrSchild
{
  public:
    //! Struct for the params of the  BH
    struct params_t
    {
        double mass = 1.0;                      //!<< The mass of the BH
        std::array<double, CH_SPACEDIM> center; //!< The center of the BH
        double spin = 0.0;                      //!< The spin param a = J / M
    };

    template <class data_t> using Vars = GHCVars::VarsWithGauge<data_t>;
    template <class data_t>
    using Diff2Vars = GHCVars::Diff2VarsWithGauge<data_t>;

    const params_t m_params;
    const double m_dx;

    KerrSchild(params_t a_params, double a_dx) : m_params(a_params), m_dx(a_dx)
    {
        // check this spin param is sensible
        if ((m_params.spin > m_params.mass) || (m_params.spin < -m_params.mass))
        {
            MayDay::Error(
                "The dimensionless spin parameter must be in the range "
                "-1.0 < spin < 1.0");
        }
    }

    /// This just calculates chi which helps with regridding, debug etc
    /// it is only done once on setup as the BG is fixed
    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // get position and set vars
        const Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        Vars<data_t> metric_vars;
	Vars<Tensor<1, data_t>> d1;
	Diff2Vars<Tensor<2, data_t>> d2;
        Tensor<4, data_t> d_chris_ULL;
        Tensor<4, data_t> Riemann;

        compute_metric_background(metric_vars, d1, d2, d_chris_ULL, Riemann, coords);

        // calculate and save chi
        //data_t chi = TensorAlgebra::compute_determinant_sym(metric_vars.gamma);
        //chi = pow(chi, -1.0 / 3.0);
        //current_cell.store_vars(chi, c_chi);
    	FOR(i) metric_vars.B[i] = 0.;
    // Populate the variables on the grid
    // NB We stil need to set Gamma^i which is NON ZERO
    // but we do this via a separate class/compute function
    // as we need the gradients of the metric which are not yet available
    	current_cell.store_vars(metric_vars);
    }

    template <class data_t> 
	    void compute_g_and_dg(Tensor<2, data_t> &g, Tensor<2, Tensor<1, data_t>> &dg,
		    	  const Coordinates<data_t> &coords) const
    {
	const double M = m_params.mass;
        const double a = m_params.spin;
        const double a2 = a * a;
	    
	// work out where we are on the grid including effect of spin
        // on x direction (length contraction)
        Tensor<1, data_t> x;
        x[0] = coords.x;
        x[1] = coords.y;
        x[2] = coords.z;
        const data_t rho = coords.get_radius();
        const data_t rho2 = rho * rho;

        // the Kerr Schild radius r
        const data_t r2 = 0.5 * (rho2 - a2) +
                          sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * x[2] * x[2]);
        const data_t r = sqrt(r2);
        const data_t cos_theta = x[2] / r;
	const data_t cos_theta2 = cos_theta * cos_theta;

        // find the H and el quantities (el decomposed into space and time)
        data_t H = M * r / (r2 + a2 * cos_theta2);

        const Tensor<1, data_t> el = {(r * x[0] + a * x[1]) / (r2 + a2),
                                      (r * x[1] - a * x[0]) / (r2 + a2), x[2] / r};
        // Calculate the gradients in el and H
        Tensor<1, data_t> dHdx;
        Tensor<2, data_t> dldx;

        using namespace TensorAlgebra;
        // derivatives of r wrt actual grid coords
        Tensor<1, data_t> drhodx;
        FOR(i) { drhodx[i] = x[i] / rho; }

        Tensor<1, data_t> drdx;
        FOR(i)
        {
            drdx[i] =
                0.5 / r *
                (rho * drhodx[i] +
                 0.5 / sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * x[2] * x[2]) *
                     (drhodx[i] * rho * (rho2 - a2) +
                      delta(i, 2) * 2.0 * a2 * x[2]));
        }

        Tensor<1, data_t> dcosthetadx;
        FOR(i) { dcosthetadx[i] = -x[2] / r2 * drdx[i] + delta(i, 2) / r; }

        FOR(i)
        {
            dHdx[i] = H * (drdx[i] / r -
                           2.0 / (r2 + a2 * cos_theta2) *
                               (r * drdx[i] + a2 * cos_theta * dcosthetadx[i]));
        }

        FOR(i)
        {
            // first the el_x comp
            dldx[0][i] =
                (x[0] * drdx[i] + r * delta(i, 0) + a * delta(i, 1) -
                 2.0 * r * drdx[i] * (r * x[0] + a * x[1]) / (r2 + a2)) /
                (r2 + a2);
            // now the el_y comp
            dldx[1][i] =
                (x[1] * drdx[i] + r * delta(i, 1) - a * delta(i, 0) -
                 2.0 * r * drdx[i] * (r * x[1] - a * x[0]) / (r2 + a2)) /
                (r2 + a2);
            // now the el_z comp
            dldx[2][i] = -x[2] * drdx[i] / r2 + delta(i, 2) / r;
        }

	FOR(i, j)
        {
            g[i][j] =
                TensorAlgebra::delta(i, j) + 2.0 * H * el[i] * el[j];
        }

        FOR(i, j, k)
        {
            dg[i][j][k] =
                2.0 * (el[i] * el[j] * dHdx[k] + H * el[i] * dldx[j][k] +
                       H * el[j] * dldx[i][k]);
        }
    }

    // Kerr Schild solution
    template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t>
    void compute_metric_background(vars_t<data_t> &vars, vars_t<Tensor<1, data_t>> &d1,
		    		   diff2_vars_t<Tensor<2, data_t>> &d2,
				   Tensor<4, data_t> &d_chris_ULL,
				   Tensor<4, data_t> &Riemann,
                                   const Coordinates<data_t> &coords) const
    {
        // black hole params - mass M and spin a
        const double M = m_params.mass;
        const double a = m_params.spin;
        const double a2 = a * a;

        // work out where we are on the grid including effect of spin
        // on x direction (length contraction)
        const data_t x = coords.x;
        const double y = coords.y;
        const double z = coords.z;
        const data_t rho = coords.get_radius();
        const data_t rho2 = rho * rho;

        // the Kerr Schild radius r
        const data_t r2 = 0.5 * (rho2 - a2) +
                          sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * z * z);
        const data_t r = sqrt(r2);
        const data_t cos_theta = z / r;

        // find the H and el quantities (el decomposed into space and time)
        data_t H = M * r / (r2 + a2 * cos_theta * cos_theta);

	const Tensor<1, data_t> el = {(r * x + a * y) / (r2 + a2),
                                      (r * y - a * x) / (r2 + a2), z / r};
        const data_t el_t = 1.0;

        // Calculate the gradients in el and H
        Tensor<1, data_t> dHdx;
	Tensor<2, data_t> d2Hdx2;
        Tensor<1, data_t> dltdx;
	Tensor<2, data_t> d2ltdx2;
        Tensor<2, data_t> dldx;
	Tensor<3, data_t> d2ldx2;
        get_KS_derivs(dHdx, d2Hdx2, dldx, d2ldx2, dltdx, d2ltdx2, H, coords);

        // populate ADM vars
        vars.lapse = pow(1.0 + 2.0 * H * el_t * el_t, -0.5);
        FOR(i, j)
        {
            vars.g[i][j] =
                TensorAlgebra::delta(i, j) + 2.0 * H * el[i] * el[j];
        }

	using namespace TensorAlgebra;
        const auto g_UU = compute_inverse_sym(vars.g);
        FOR(i)
        {
            vars.shift[i] = 0.;
            FOR(j)
            {
                vars.shift[i] += g_UU[i][j] * 2.0 * H * el[j] * el_t;
            }
        }

        // Calculate partial derivative of spatial metric
        FOR(i, j, k)
        {
            d1.g[i][j][k] =
                2.0 * (el[i] * el[j] * dHdx[k] + H * el[i] * dldx[j][k] +
                       H * el[j] * dldx[i][k]);
        }

	FOR(i, j, k, l)
	{
	    d2.g[i][j][k][l] =
                2.0 * (dldx[i][l] * el[j] * dHdx[k] + el[i] * dldx[j][l] * dHdx[k] +
		       el[i] * el[j] * d2Hdx2[k][l] + dHdx[l] * el[i] * dldx[j][k] +
                       H * dldx[i][l] * dldx[j][k] + H * el[i] * d2ldx2[j][k][l] + 
		       dHdx[l] * el[j] * dldx[i][k] + H * dldx[j][l] * dldx[i][k] +
		       H * el[j] * d2ldx2[i][k][l]);
	}

        // calculate derivs of lapse and shift
        FOR(i)
        {
            d1.lapse[i] = -pow(vars.lapse, 3.0) * el_t *
                               (el_t * dHdx[i] + 2.0 * H * dltdx[i]);
        }

	FOR(i,j)
	{
	    d2.lapse[i][j] = 
		-pow(vars.lapse, 2.0) * 
			(3.0 * el_t * d1.lapse[j] *
				(el_t * dHdx[i] + 2.0 * H * dltdx[i]) +
			 vars.lapse * 
			 	(dltdx[j] * (el_t * dHdx[i] + 2.0 * H * dltdx[i]) +
			       	 el_t * (dltdx[j] * dHdx[i] + el_t * d2Hdx2[i][j] + 
				 	2.0 * dHdx[j] * dltdx[i] + 2.0 * H * d2ltdx2[i][j])));		;
	}

        // use the fact that shift^i = lapse^2 * shift_i
        FOR(i, j)
        {
            d1.shift[i][j] =
                2.0 * el_t * dHdx[j] * pow(vars.lapse, 2.0) * el[i] +
                4.0 * el_t * H * vars.lapse * d1.lapse[j] * el[i] +
                2.0 * el_t * H * pow(vars.lapse, 2.0) * dldx[i][j] +
                2.0 * dltdx[j] * H * pow(vars.lapse, 2.0) * el[i];
        }

	FOR(i, j, k)
	{
	    d2.shift[i][j][k] = 2.0 * dltdx[k] * dHdx[j] * pow(vars.lapse, 2.0) * el[i] +
		    		2.0 * el_t * d2Hdx2[j][k] * pow(vars.lapse, 2.0) * el[i] +
				4.0 * el_t * dHdx[j] * vars.lapse * d1.lapse[k] * el[i] +
				2.0 * el_t * dHdx[j] * pow(vars.lapse, 2.0) * dldx[i][k] +
				4.0 * dltdx[k] * H * vars.lapse * d1.lapse[j] * el[i] +
				4.0 * el_t * dHdx[k] * vars.lapse * d1.lapse[j] * el[i] +
				4.0 * el_t * H * d1.lapse[k] * d1.lapse[j] * el[i] +
				4.0 * el_t * H * vars.lapse * d2.lapse[j][k] * el[i] +
				4.0 * el_t * H * vars.lapse * d1.lapse[j] * dldx[i][k] +
				2.0 * dltdx[k] * H * pow(vars.lapse, 2.0) * dldx[i][j] +
				2.0 * el_t * dHdx[k] * pow(vars.lapse, 2.0) * dldx[i][j] +
				4.0 * el_t * H * vars.lapse * d1.lapse[k] * dldx[i][j] +
				2.0 * el_t * H * pow(vars.lapse, 2.0) * d2ldx2[i][j][k] +
				2.0 * d2ltdx2[j][k] * H * pow(vars.lapse, 2.0) * el[i] +
				2.0 * dltdx[j] * dHdx[k] * pow(vars.lapse, 2.0) * el[i] +
				4.0 * dltdx[j] * H * vars.lapse * d1.lapse[k] * el[i] +
				2.0 * dltdx[j] * H * pow(vars.lapse, 2.0) * dldx[i][k];
	}

        // calculate the extrinsic curvature, using the fact that
        // 2 * lapse * K_ij = D_i \beta_j + D_j \beta_i - dgamma_ij dt
        // and dgamma_ij dt = 0 in chosen fixed gauge
        const auto chris = compute_christoffel(d1.g, g_UU);
	//FOR(i) vars.Gam[i] = chris.contracted[i];
        FOR(i, j)
        {
            vars.K[i][j] = 0.0;
            FOR(k)
            {
                vars.K[i][j] +=
                    vars.g[k][j] * d1.shift[k][i] +
                    vars.g[k][i] * d1.shift[k][j] +
                    (d1.g[k][i][j] + d1.g[k][j][i]) *
                        vars.shift[k];
                FOR(m)
                {
                    vars.K[i][j] += -2.0 * chris.ULL[k][i][j] *
                                           vars.g[k][m] * vars.shift[m];
                }
            }
            vars.K[i][j] *= 0.5 / vars.lapse;
        }
	vars.Pi = -compute_trace(vars.K, g_UU);

	FOR(i, j, k, l)
	{
	   d_chris_ULL[i][j][k][l] = 0.;
	   FOR(m)
	   {
	      d_chris_ULL[i][j][k][l] += 0.5 * g_UU[i][m] *
		      	(d2.g[k][m][j][l] + d2.g[j][m][k][l] - d2.g[j][k][m][l]);
	      FOR(p, q)
	      {
		 d_chris_ULL[i][j][k][l] -= g_UU[i][p] * g_UU[m][q] *
			 d1.g[p][q][l] * chris.LLL[m][j][k];
	      }
	   }
	}

	FOR(i, j, k, l)
	{
	    Riemann[i][j][k][l] = d_chris_ULL[i][l][j][k] - d_chris_ULL[i][k][j][l];
	    FOR(m)
	    {
		Riemann[i][j][k][l] += chris.ULL[i][k][m] * chris.ULL[m][l][j] -
				       chris.ULL[i][l][m] * chris.ULL[m][k][j];
	    }

	}
	
	FOR(i, j, k)
        {
            d1.K[i][j][k] = 0.0;
            FOR(l)
            {
                d1.K[i][j][k] +=
                    d1.g[l][j][k] * d1.shift[l][i] +
		    vars.g[l][j] * d2.shift[l][i][k] +
                    d1.g[l][i][k] * d1.shift[l][j] +
		    vars.g[l][i] * d2.shift[l][j][k] +
                    (d2.g[l][i][j][k] + d2.g[l][j][i][k]) *
                        vars.shift[l] +
		    (d1.g[l][i][j] + d1.g[l][j][i]) *
                        d1.shift[l][k];
                FOR(m)
                {
                    d1.K[i][j][k] += -2.0 * chris.ULL[l][i][j] *
                                           (d1.g[l][m][k] * vars.shift[m] +
					    vars.g[l][m] * d1.shift[m][k]) -
				     2.0 * d_chris_ULL[l][i][j][k] *
				     	   vars.g[l][m] * vars.shift[m];
                }
            }
            d1.K[i][j][k] *= 0.5 / vars.lapse;
	    d1.K[i][j][k] -= vars.K[i][j] * d1.lapse[k] / vars.lapse;
        }
    }

  protected:
    /// Work out the gradients of the quantities H and el appearing in the Kerr
    /// Schild solution
    template <class data_t>
    void get_KS_derivs(Tensor<1, data_t> &dHdx, Tensor<2, data_t> &d2Hdx2, 
		       Tensor<2, data_t> &dldx, Tensor<3, data_t> &d2ldx2, 
		       Tensor<1, data_t> &dltdx, Tensor<2, data_t> &d2ltdx2, 
		       const data_t &H,
                       const Coordinates<data_t> &coords) const
    {
        // black hole params - spin a
        const double a = m_params.spin;
        const double a2 = a * a;

        // work out where we are on the grid, and useful quantities
        Tensor<1, data_t> x;
        x[0] = coords.x;
        x[1] = coords.y;
        x[2] = coords.z;
        const double z = coords.z;
        const data_t rho = coords.get_radius();
        const data_t rho2 = rho * rho;

        // the Kerr Schild radius r
        const data_t r2 = 0.5 * (rho2 - a2) +
                          sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * z * z);
        const data_t r = sqrt(r2);
        const data_t cos_theta = z / r;
        const data_t cos_theta2 = cos_theta * cos_theta;

        using namespace TensorAlgebra;
        // derivatives of r wrt actual grid coords
        Tensor<1, data_t> drhodx;
        FOR(i) { drhodx[i] = x[i] / rho; }

	Tensor<2, data_t> d2rhodx2;
	FOR(i,j) d2rhodx2[i][j] = (delta(j, i) - drhodx[j] * x[i] / rho) / rho;

        Tensor<1, data_t> drdx;
        FOR(i)
        {
            drdx[i] =
                0.5 / r *
                (rho * drhodx[i] +
                 0.5 / sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * z * z) *
                     (drhodx[i] * rho * (rho2 - a2) +
                      delta(i, 2) * 2.0 * a2 * z));
        }

	Tensor<2, data_t> d2rdx2;
	FOR(i,j)
	{
	    d2rdx2[i][j] = -0.5 * drdx[j] / r2 *
		    	   (rho * drhodx[i] +
                 	    0.5 / sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * z * z) *
                     		(drhodx[i] * rho * (rho2 - a2) +
                      		 delta(i, 2) * 2.0 * a2 * z)) +
			   0.5 / r *
			   (drhodx[j] * drhodx[i] + rho * d2rhodx2[i][j] +
			    0.5 / sqrt(0.25 * (rho2 - a2) * (rho2 - a2) + a2 * z * z) *
			   	(d2rhodx2[i][j] * rho * (rho2 - a2) +
				 drhodx[i] * drhodx[j] * (rho2 - a2) +
				 2.0 * drhodx[i] * rho * rho * drhodx[j] +
				 2.0 * delta(i, 2) * a2 * delta(j, 2) -
				 0.5 / (0.25 * (rho2 - a2) * (rho2 - a2) + a2 * z * z) *
				     (rho * drhodx[j] * (rho2 - a2) + delta(j, 2) * 2.0 * a2 * z) *
				     (rho * drhodx[i] * (rho2 - a2) + delta(i, 2) * 2.0 * a2 * z)));
	}

        Tensor<1, data_t> dcosthetadx;
        FOR(i) { dcosthetadx[i] = -z / r2 * drdx[i] + delta(i, 2) / r; }

	Tensor<2, data_t> d2costhetadx2;
	FOR(i,j) d2costhetadx2[i][j] = -(delta(j, 2) * drdx[i] + x[2] * d2rdx2[i][j] - 
						x[2] * 2.0 * r * drdx[i] * drdx[j] / r2) / r2 -
					delta(i, 2) * drdx[j] / r2;

        FOR(i)
        {
            dHdx[i] = H * (drdx[i] / r -
                           2.0 / (r2 + a2 * cos_theta2) *
                               (r * drdx[i] + a2 * cos_theta * dcosthetadx[i]));
        }

	FOR(i,j)
	{
	    d2Hdx2[i][j] = dHdx[j] * (drdx[i] / r -
                           2.0 / (r2 + a2 * cos_theta2) *
                               (r * drdx[i] + a2 * cos_theta * dcosthetadx[i])) +
		    	   H * (d2rdx2[i][j] / r - drdx[i] * drdx[j] / r2 -
				2.0 / (r2 + a2 * cos_theta2) *
				    (drdx[j] * drdx[i] + r * d2rdx2[i][j] +
				     a2 * dcosthetadx[j] * dcosthetadx[i] +
				     a2 * cos_theta * d2costhetadx2[i][j] -
				     2.0 * (r * drdx[i] + a2 * cos_theta * dcosthetadx[i]) *
				    	   (r * drdx[j] + a2 * cos_theta * dcosthetadx[j]) / 
					   	(r2 + a2 * cos_theta)));
	}

        // note to use convention as in rest of tensors the last index is the
        // derivative index so these are d_i l_j
        FOR(i)
        {
            // first the el_x comp
            dldx[0][i] =
                (x[0] * drdx[i] + r * delta(i, 0) + a * delta(i, 1) -
                 2.0 * r * drdx[i] * (r * x[0] + a * x[1]) / (r2 + a2)) /
                (r2 + a2);
            // now the el_y comp
            dldx[1][i] =
                (x[1] * drdx[i] + r * delta(i, 1) - a * delta(i, 0) -
                 2.0 * r * drdx[i] * (r * x[1] - a * x[0]) / (r2 + a2)) /
                (r2 + a2);
            // now the el_z comp
            dldx[2][i] = -x[2] * drdx[i] / r2 + delta(i, 2) / r;
        }

	FOR(i,j)
	{
	    d2ldx2[0][i][j] = 
		(delta(j, 0) * drdx[i] + x[0] * d2rdx2[i][j] + drdx[j] * delta(i, 0) - 
		 	2.0 * ((drdx[j] * drdx[i] + r * d2rdx2[i][j]) * (r * x[0] + a * x[1]) +
		 		r * drdx[i] * (drdx[j] * x[0] + r * delta(j, 0) + 
					a * delta(j, 1))) / (r2 + a2) -
		 	2.0 * r * drdx[j] * (x[0] * drdx[i] + r * delta(i, 0) + a * delta(i, 1) - 
		  			4.0 * r * drdx[i] * (r * x[0] + a * x[1]) / (r2 + a2)) / (r2 + a2)) / 
		(r2 + a2);
	    d2ldx2[1][i][j] =
                (delta(j, 1) * drdx[i] + x[1] * d2rdx2[i][j] + drdx[j] * delta(i, 1) -
                        2.0 * ((drdx[j] * drdx[i] + r * d2rdx2[i][j]) * (r * x[1] - a * x[0]) +
                                r * drdx[i] * (drdx[j] * x[1] + r * delta(j, 1) -
                                        a * delta(j, 0))) / (r2 + a2) -
                        2.0 * r * drdx[j] * (x[1] * drdx[i] + r * delta(i, 1) - a * delta(i, 0) -
                                        4. * r * drdx[i] * (r * x[1] - a * x[0]) / (r2 + a2)) / (r2 + a2)) /
                (r2 + a2);
	    d2ldx2[2][i][j] = -(delta(j, 2) * drdx[i] + x[2] * d2rdx2[i][j] - 
			    		2.0 * r * drdx[j] * x[2] * drdx[i] / r2) / r2
		    	      - delta(i, 2) * drdx[j] / r2;
	}

        // then dltdx
        FOR(i) { 
	   dltdx[i] = 0.0;
	   FOR(j) d2ltdx2[i][j] = 0.0; 
	}
    }

  public:
    // used to decide when to excise - ie when within the horizon of the BH
    // note that this is not templated over data_t
    bool check_if_excised(const Coordinates<double> &coords) const
    {
        // black hole params - mass M and spin a
        const double M = m_params.mass;
        const double a = m_params.spin;
        const double a2 = a * a;

        // work out where we are on the grid
        const double x = coords.x;
        const double y = coords.y;
        const double z = coords.z;
        const double r_plus = M + sqrt(M * M - a2);
        const double r_minus = M - sqrt(M * M - a2);

        // position relative to outer horizon - 1 indicates on horizon
        // less than one is within
        const double outer_horizon =
            (x * x + y * y) / (2.0 * M * r_plus) + z * z / r_plus / r_plus;

        // position relative to inner horizon - 1 indicates on horizon, less
        // than 1 is within
        const double inner_horizon =
            (x * x + y * y) / (2.0 * M * r_minus) + z * z / r_minus / r_minus;

        bool is_excised = false;
        // value less than 1 indicates we are within the horizon
        if (outer_horizon < 0.9 || inner_horizon < 1.05)
        {
            is_excised = true;
        }
        return is_excised;
    }
};

#endif /* KERRSCHILD_HPP_ */

