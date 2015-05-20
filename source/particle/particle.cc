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

#include <aspect/particle/particle.h>

namespace aspect
{
  namespace Particle
  {
    template <int dim>
    inline
    BaseParticle<dim>::BaseParticle (const Point<dim> &new_loc,
                                     const double &new_id)
      :
      location (new_loc),
      id (new_id),
      is_local (true),
      check_vel (true)
    {
    }

    template <int dim>
    inline
    BaseParticle<dim>::BaseParticle ()
      :
      location (),
      velocity (),
      lastLocation (),
      lastVelocity (),
      id (0),
      is_local (true),
      check_vel (true)
    {
    }


    template <int dim>
    inline
    BaseParticle<dim>::~BaseParticle ()
    {
    }


    template <int dim>
    void
    BaseParticle<dim>::set_location (const Point<dim> &new_loc)
    {
      location = new_loc;
    }

    template <int dim>
    void
    BaseParticle<dim>::set_lastLocation (const Point<dim> &new_loc)
    {
      lastLocation = new_loc;
    }


    template <int dim>
    void
    BaseParticle<dim>::write_data (std::vector<double> &data) const
    {
      // Write location data
      for (unsigned int i = 0; i < dim; ++i)
        {
          data.push_back(location(i));
        }
      // Write velocity data
      for (unsigned int i = 0; i < dim; ++i)
        {
          data.push_back(velocity(i));
        }
      // Write old location data
      for (unsigned int i = 0; i < dim; ++i)
        {
          data.push_back(lastLocation(i));
        }
      // Write old velocity data
      for (unsigned int i = 0; i < dim; ++i)
        {
          data.push_back(lastVelocity(i));
        }
      data.push_back(id);
    }

    template <int dim>
    unsigned int BaseParticle<dim>::read_data(const std::vector<double> &data, const unsigned int &pos)
    {
      unsigned int p = pos;
      // Read location data
      for (unsigned int i = 0; i < dim; ++i)
        {
          location (i) = data[p++];
        }
      // Write velocity data
      for (unsigned int i = 0; i < dim; ++i)
        {
          velocity (i) = data[p++];
        }
      // Read old location data
      for (unsigned int i = 0; i < dim; ++i)
        {
          lastLocation (i) = data[p++];
        }
      // Write old velocity data
      for (unsigned int i = 0; i < dim; ++i)
        {
          lastVelocity (i) = data[p++];
        }
      id = data[p++];
      return p;
    }

    template <int dim>
    unsigned int
    BaseParticle<dim>::data_len ()
    {
      return (dim + dim + dim + dim + 1);
    }


    template <int dim>
    Point<dim>
    BaseParticle<dim>::get_location () const
    {
      return location;
    }

    template <int dim>
    Point<dim>
    BaseParticle<dim>::get_lastLocation () const
    {
      return lastLocation;
    }

    template <int dim>
    void
    BaseParticle<dim>::set_velocity (Point<dim> new_vel)
    {
      velocity = new_vel;
    }

    template <int dim>
    void
    BaseParticle<dim>::set_lastVelocity (Point<dim> new_vel)
    {
      lastVelocity = new_vel;
    }

    template <int dim>
    Point<dim>
    BaseParticle<dim>::get_velocity () const
    {
      return velocity;
    }

    template <int dim>
    Point<dim>
    BaseParticle<dim>::get_lastVelocity () const
    {
      return lastVelocity;
    }

    template <int dim>
    double
    BaseParticle<dim>::get_id () const
    {
      return id;
    }

    template <int dim>
    bool
    BaseParticle<dim>::local () const
    {
      return is_local;
    }

    template <int dim>
    void
    BaseParticle<dim>::set_local (bool new_local)
    {
      is_local = new_local;
    }

    template <int dim>
    bool
    BaseParticle<dim>::vel_check () const
    {
      return check_vel;
    }

    template <int dim>
    void
    BaseParticle<dim>::set_vel_check (bool new_vel_check)
    {
      check_vel = new_vel_check;
    }

    template <int dim>
    void
    BaseParticle<dim>::add_mpi_types (std::vector<MPIDataInfo> &data_info)
    {
      // Add the position, velocity, ID
      data_info.push_back (
        MPIDataInfo ("pos", dim));
      data_info.push_back (
        MPIDataInfo ("velocity", dim));
      // Add the old position, old velocity
      data_info.push_back (
        MPIDataInfo ("oldpos", dim));
      data_info.push_back (
        MPIDataInfo ("oldvelocity", dim));
      data_info.push_back (MPIDataInfo ("id", 1));
    }



    // explicit instantiation
    template class BaseParticle<2>;
    template class BaseParticle<3>;
  }
}
