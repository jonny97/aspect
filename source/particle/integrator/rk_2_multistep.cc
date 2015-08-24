/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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

#include <aspect/particle/integrator/rk_2_multistep.h>
#include <aspect/postprocess/tracer.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace Particle
  {
    namespace Integrator
    {
      template <int dim>
      RK2IntegratorMultiStep<dim>::RK2IntegratorMultiStep()
      {
        step = 0;
      }

      template <int dim>
      bool
      RK2IntegratorMultiStep<dim>::integrate_step(typename std::multimap<LevelInd, Particle<dim> > &particles,
                                                  const std::vector<Tensor<1,dim> > &old_velocities,
                                                  const std::vector<Tensor<1,dim> > &velocities,
                                                  const double dt)
      {
        const Postprocess::PassiveTracers<dim> *tracer_postprocess =
          this->template find_postprocessor<const Postprocess::PassiveTracers<dim> >();
        Assert(tracer_postprocess!=NULL,
               ExcMessage("The RK2IntegratorMultiStep queried the tracer postprocessor, "
                          "but did not find it."));

        const Property::Manager<dim> manager = tracer_postprocess->get_particle_world().get_manager();

        const unsigned int last_velocity_component = manager.get_property_component_by_name("velocity");
        const unsigned int last_position_component = manager.get_property_component_by_name("position");

        typename std::multimap<LevelInd, Particle<dim> >::iterator it=particles.begin();
        typename std::vector<Tensor<1,dim> >::const_iterator old_vel = old_velocities.begin();
        typename std::vector<Tensor<1,dim> >::const_iterator vel = velocities.begin();

        for (; it!=particles.end(), vel!=velocities.end(), old_vel!=old_velocities.end(); ++it,++vel,++old_vel)
          {
            const std::vector<double> data = it->second.get_properties();

            Point<dim> lastLoc;
            Tensor<1,dim> lastVel;
            for (unsigned int i = 0; i < dim; ++i)
              {
                lastLoc(i) = data[last_position_component + i];
                lastVel[i] = data[last_velocity_component + i];
              }

            if (step == 0)
              {
                it->second.set_location(lastLoc + 0.5 * dt * lastVel);
              }
            else if (step == 1)
              {
                it->second.set_location(lastLoc + dt*(*vel + *old_vel) / 2.0);
              }
            else
              {
                Assert(false,
                       ExcMessage("The RK2 integrator should never continue after two integration steps."));
              }
          }

        step = (step+1)%2;

        // Continue until we're at the last step
        return (step != 0);
      }

      template <int dim>
      unsigned int
      RK2IntegratorMultiStep<dim>::data_length() const
      {
        return 0;
      }

      template <int dim>
      void
      RK2IntegratorMultiStep<dim>::read_data(std::vector<double>::const_iterator &data,
                                             const double &id_num)
      {
      }

      template <int dim>
      void
      RK2IntegratorMultiStep<dim>::write_data(std::vector<double>::iterator &data,
                                              const double &id_num) const
      {
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Integrator
    {
      ASPECT_REGISTER_PARTICLE_INTEGRATOR(RK2IntegratorMultiStep,
                                          "rk2 multistep",
                                          "Runge Kutta second order integrator, where "
                                          "y_{n+1} = y_n + dt*v(0.5*k_1), k_1 = dt*v(y_n). "
                                          "This scheme is identical to the rk2 integrator, "
                                          "but instead of storing the old location in the "
                                          "integrator, it is stored as particle property.")
    }
  }
}
