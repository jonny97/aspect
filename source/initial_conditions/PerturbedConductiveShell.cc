
/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/
/*  $Id: box.cc 878 2012-04-03 10:28:04Z bangerth $  */


#include <aspect/initial_conditions/perturbed_spherical_shell.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <fstream>
#include <iostream>
#include <cstring>

namespace aspect
{
	namespace InitialConditions
	{
		template <int dim>
		double
		PerturbedConductiveShell<dim>::
		initial_temperature (const Point<dim> &position) const
		{
			// this initial condition only makes sense if the geometry is a
			// spherical shell. verify that it is indeed
			AssertThrow (dynamic_cast<const GeometryModel::SphericalShell<dim>*>
						 (&this->get_geometry_model())
						 != 0,
						 ExcMessage ("This initial condition can only be used if the geometry "
									 "is a spherical shell."));
		
		// Inner radius
		const double Rin = dynamic_cast<const GeometryModel::SphericalShell<dim>&> (this->get_geometry_model()).inner_radius();
		// Outer radius
		const double Rout = dynamic_cast<const GeometryModel::SphericalShell<dim>&> (this->get_geometry_model()).outer_radius();
		// Temperature on the inner boundary
		const double inner_temperature = this->get_boundary_temperature().maximal_temperature();
		// temperature on the outer boundary
		const double outer_temperature = this->get_boundary_temperature().minimal_temperature();
		// Delta T		
		const double dT = inner_temperature - outer_temperature;
		// Radius at position
		const double radius = position.norm();
		
		double angle, cosfi, sinfi, costh, sinth, temperature, Yl2m2, Yl1m1, coef;
		
		if (dim==2){
			// Trigonometry
			// longitude
			angle = std::atan2(position[1],position[0]);
			cosfi = std::cos(angle);
			sinfi = std::sin(angle);
			
			// Conductive profile
			temperature = Rin*(radius-Rout)/radius/(Rin-Rout);
			
			// Initial perturbation : cubic
			coef= 3.0 * std::sqrt(35.0)/16.0/std::sqrt(numbers::PI);
			Yl2m2 = coef * (std::pow(cosfi,4) - 6*std::pow(cosfi,2)*std::pow(sinfi,2) + std::pow(sinfi,4));
			temperature+= amplitude * Yl2m2 * std::sin(numbers::PI* (radius-Rin)/(Rout-Rin));
		}
		if (dim==3) {
			// Trigonometry
			//longitude
			angle = std::atan2(position[1],position[0]);
			cosfi = std::cos(angle);
			sinfi = std::sin(angle);
			//colatitude
			angle = std::acos(position[2]/radius)-numbers::PI/2.0;
			costh = std::cos(angle);
			sinth = std::sin(angle);
			
			// Conductive profile
			temperature = Rin*(radius-Rout)/radius/(Rin-Rout);
			
			switch (mode) {
				case 32:
						// Initial perturbation : tetrahedral
						// l=3 and m=2
					coef= 15.0*std::sqrt(7.0/240.0/numbers::PI);
					// cos(2*phi)=1-2*sin^2(phi)
					// sin(2*phi)=2*sin(phi)*cos(phi)
					Yl1m1=coef*sinth*std::pow(costh,2)*(1-2.0*std::pow(sinfi,2)+2.0*sinfi*cosfi);
					Yl2m2=0;
					break;

				case 44:
					// Initial perturbation : cubic
					// l=4 and m=0
					coef= 3.0 / 16.0 / std::sqrt(numbers::PI);
					Yl1m1 = coef * (35.0 * std::pow(sinth,4) - 30.0 * std::pow(sinth,2) +3);
					// l=4 and m=4
					coef= 3.0 * std::sqrt(35.0)/16.0/std::sqrt(numbers::PI);
					Yl2m2 = 5.0/7.0 * (1.0-delta)* coef * (std::pow(cosfi*costh,4) - 6*std::pow(cosfi*costh,2)*std::pow(sinfi*costh,2) + std::pow(sinfi*costh,4));
					break;
				case 66:
					// Initial perturbation : dodecahedron
					coef= std::sqrt(14.0/11.0);
					// l=6 and m=0
					Yl1m1 = 1.0/32.0 * std::sqrt(13.0/numbers::PI) * (231.0*std::pow(sinth,6.0) - 315.0*std::pow(sinth,4.0)+105.0*std::pow(sinth,2.0)-5.0);
					// l=6 and m=5
					Yl2m2 = coef* 3.0/16.0 * std::sqrt(1001.0/2.0/numbers::PI) * sinth *(std::pow(costh*cosfi,5.0) - 10.0*pow(costh*sinfi,2.0)*pow(costh*cosfi,3.0)+5.0*pow(sinfi*costh,4.0)*(cosfi*costh));
					break;
				case 34:
					// Initial perturbation : hexahedron
					// l=4 and m=0
					coef= 3.0 / 16.0 / std::sqrt(numbers::PI);
					Yl1m1 = coef * (35.0 * std::pow(sinth,4) - 30.0 * std::pow(sinth,2) +3);
					// l=3 and m=3
					coef= std::sqrt(7.0/12.0/numbers::PI);
					Yl2m2 = delta * coef * ( -15 * std::pow(costh,3) * std::cos(angle*3));
					break;
				case 43:
					// Initial perturbation : pentahedron
					// l=3 and m=0
					coef=1.0/2.0;
					Yl1m1 = coef * (5.0 * std::pow(costh,3) - 3 * sinth);				
					// l=4 and m=4
					coef= 3.0 * std::sqrt(35.0)/16.0/std::sqrt(numbers::PI);
					Yl2m2 = delta * coef * (std::pow(cosfi*costh,4) - 6*std::pow(cosfi*costh,2)*std::pow(sinfi*costh,2) + std::pow(sinfi*costh,4));
					break;
			}
			// add perturbation to conductive temperature
			temperature += amplitude * (Yl1m1 + Yl2m2) * std::sin(numbers::PI* (radius-Rin)/(Rout-Rin));	

		}
		
		// Unnormalized temperature
		temperature*= dT;
		temperature+= outer_temperature;
		
		return temperature;
    }
	  
	  template <int dim>
	  void
	  PerturbedConductiveShell<dim>::declare_parameters (ParameterHandler &prm)
	  {
		  
		  prm.enter_subsection("Initial conditions");
		  {
			  prm.enter_subsection("Perturbed conductive shell");
			  {
				  prm.declare_entry ("Amplitude", "0.01",
									 Patterns::Double (),
									 "amplitude of anomaly.");
				  prm.declare_entry ("delta", "0.0", Patterns::Double (), "additional perturbation.");
				  prm.declare_entry ("mode", "44", Patterns::Double (), "initial symmetry.");
			  }
			  prm.leave_subsection ();
		  }
		  prm.leave_subsection ();
	  }
	  
	  
	  template <int dim>
	  void
	  PerturbedConductiveShell<dim>::parse_parameters (ParameterHandler &prm)
	  {
		  
		  prm.enter_subsection("Initial conditions");
		  {
			  prm.enter_subsection("Perturbed conductive shell");
			  {
				  amplitude = prm.get_double ("Amplitude");
				  delta     = prm.get_double ("delta");
				  mode		= prm.get_integer ("mode");
			  }
			  prm.leave_subsection ();
		  }
		  prm.leave_subsection ();
	  }	  
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialConditions
  {
    ASPECT_REGISTER_INITIAL_CONDITIONS(PerturbedConductiveShell,
                                       "Perturbed conductive shell",
                                       "An initial temperature field in which the temperature "
                                       "is linear from top to bottom and is perturbed slightly."
									   " 32 == tetrahedral initial perturbation"
									   " 44 == cubic initial perturbation"
									   " 66 == dodecahedron initial condition"
									   " 34 == pentahedron initial condition (5cells)"
									   " 43 == hexahedron initial condition (5cells)")
  }
}
