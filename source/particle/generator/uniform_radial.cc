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

#include <aspect/particle/generator/uniform_radial.h>

#include <boost/random.hpp>


namespace aspect
{
  namespace Particle
  {
    namespace Generator
    {
      // Generate a uniform radial distribution of particles over the specified region
      // in the computational domain

          /**
           * Constructor.
           *
           * @param[in] The MPI communicator for synchronizing particle generation.
           */
        template <int dim>
        UniformRadial<dim>::UniformRadial() {}

          /**
           * Generate a uniformly randomly distributed set of particles in the current triangulation.
           */
          // TODO: fix the particle system so it works even with processors assigned 0 cells
        template <int dim>
        void
		UniformRadial<dim>::generate_particles(Particle::World<dim> &world,
                                                        const double total_num_particles)
        {
          double      total_volume, local_volume, subdomain_fraction, start_fraction, end_fraction;

          // Calculate the number of particles in this domain as a fraction of total volume
          total_volume = local_volume = 0;
          for (typename parallel::distributed::Triangulation<dim>::active_cell_iterator
               it=this->get_triangulation().begin_active();
               it!=this->get_triangulation().end(); ++it)
            {
              double cell_volume = it->measure();
              AssertThrow (cell_volume != 0, ExcMessage ("Found cell with zero volume."));

              if (it->is_locally_owned())
                local_volume += cell_volume;
            }
          // Sum the local volumes over all nodes
          MPI_Allreduce(&local_volume, &total_volume, 1, MPI_DOUBLE, MPI_SUM, this->get_mpi_communicator());

          // Assign this subdomain the appropriate fraction
          subdomain_fraction = local_volume/total_volume;

          // Sum the subdomain fractions so we don't miss particles from rounding and to create unique IDs
          MPI_Scan(&subdomain_fraction, &end_fraction, 1, MPI_DOUBLE, MPI_SUM, this->get_mpi_communicator());
          start_fraction = end_fraction-subdomain_fraction;

          // Calculate start and end IDs so there are no gaps
          // TODO: this can create gaps for certain processor counts because of
          // floating point imprecision, figure out how to fix it
          const unsigned int  start_id = static_cast<unsigned int>(std::ceil(start_fraction*total_num_particles));
          const unsigned int  end_id   = static_cast<unsigned int>(fmin(std::ceil(end_fraction*total_num_particles), total_num_particles));
          const unsigned int  subdomain_particles = end_id - start_id;

          uniformly_distributed_particles_in_subdomain(world, subdomain_particles, start_id);
        }

          /**
           * Generate a set of particles uniformly distributed within the
           * specified region.  We do cell-by-cell assignment of
           * particles because the decomposition of the mesh may result in a highly
           * non-rectangular local mesh which makes uniform particle distribution difficult.
           *
           * @param [in] world The particle world the particles will exist in
           * @param [in] num_particles The number of particles to generate in this subdomain
           * @param [in] start_id The starting ID to assign to generated particles
           */
        template <int dim>
          void
          UniformRadial<dim>::uniformly_distributed_particles_in_subdomain (Particle::World<dim> &world,
                                                      const unsigned int num_particles,
                                                      const unsigned int start_id)
          {
            unsigned int cur_id;
            cur_id = start_id;
            //[radiusMin, radiusMax, thetaMin, thetaMax, phiMin, phiMax]
            //int shellCount = limits[0];

            if (dim == 3)
              {
                double thetaSeperation, phiSeperation;
                for (int i = 0; i < shellCount; i++)
                  {
                    int thetaParticles, phiParticles;
                    thetaParticles = floor(sqrt(particlesPerRadius[i]));
                    phiParticles = particlesPerRadius[i] / thetaParticles;
                    //cout << "shell: " << i << "\nTheta Particles: " << thetaParticles << "\nPhi   Particles: " << phiParticles << "\n";
                    //thetaSeperation = (limits[4] - limits[3]) / thetaParticles;
                    phiSeperation = (limits[6] - limits[5]) / phiParticles;
                    int *ppPh = new int[phiParticles];
                    double phiTotalLength = 0;
                    int j = 0;

                    for (double phi = limits[5]; phi < limits[6]; phi += phiSeperation, j++)
                      {
                        //if (j > phiParticles)
                        //{
                        //  cout << "Error! j > thetaParticles!\n";
                        //  break;
                        //}
                        //Average value of sin(n) from 0 to 180 degrees is (2/pi)
                        ppPh[j] = (thetaParticles * sin(phi / 180 * M_PI) * (M_PI / 2)) + 1;
                        //cout << "Test1: " << ppPh[j] << "\n";
                      }

                    j = 0;
                    for (double phi = limits[5]; phi < limits[6]; phi += phiSeperation, j++)
                      {
                        //if (j > phiParticles)
                        //{
                        //  cout << "Error 2! j > thetaParticles!\n";
                        //  break;
                        //}
                        thetaSeperation = (limits[4] - limits[3]) / ppPh[j];
                        //cout << "Test Theta: " << thetaSeperation << "\n";
                        for (double theta = limits[3]; theta < limits[4]; theta += thetaSeperation)
                          {
                            //cout << "test1\n";
                            Point<dim> newPoint
                            (
                              shell[i] * sin(phi / 180 * M_PI) * cos(theta / 180 * M_PI),
                              shell[i] * sin(phi / 180 * M_PI) * sin(theta / 180 * M_PI),
                              shell[i] * cos(phi / 180 * M_PI)
                            );
                            //cout << "test2: " << newPoint << "\n";
                            cur_id++;
                            try
                              {
                                //std::cout << "Making a point: <" << newPoint << ">\n" << "Spherical coords: [" << shell[i] << ", " << theta << ", " << phi << "]\n";

                                //Modify the find_active_cell_around_point to only search for nearest vertex  and adj. cells, instead of searching all cells in the simulation

                                //cout << "test3.1\n";
                                typename parallel::distributed::Triangulation<dim>::active_cell_iterator it =
                                  (GridTools::find_active_cell_around_point<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                                  //(GridTools::find_active_cell_around_point_quick<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                                //std::cout << "Point successful!\n";
                                //cout << "test3.2\n";

                                if (it->is_locally_owned())
                                  {
                                    //Only try to add the point if the cell it is in, is on this processor
                                    //cout << "test3.2.1\n";
                                    T new_particle(newPoint, cur_id);
                                    //cout << "test3.2.2\n";
                                    world.add_particle(new_particle, std::make_pair(it->level(), it->index()));
                                    //cout << "test3.2.3\n";
                                  }
                                //cout << "test3.3\n";
                              }
                            catch (...)
                              {
                                //cout << "test3.Catch\n";
                                //A point wasn't in an available cell, this might be because it isn't local; we can ignore it

                                //std::cout << "Point failed.\n";
                                //Allow this loss for now
                                //AssertThrow (false, ExcMessage ("Couldn't generate particle (Shell too close to boundary?)."));
                              }
                            //cout << "test4\n";
                          }
                      }
                    delete ppPh;
                  }
              }
            else if (dim == 2)
              {
                double thetaSeperation, phiSeperation;
                for (int i = 0; i < shellCount; i++)
                  {
                    int thetaParticles, phiParticles;
                    thetaParticles = particlesPerRadius[i];
                    thetaSeperation = (limits[4] - limits[3]) / thetaParticles;
                    for (double theta = limits[3]; theta < limits[4]; theta += thetaSeperation)
                      {
                        Point<dim> newPoint
                        (
                          shell[i] * cos(theta / 180 * M_PI),
                          shell[i] * sin(theta / 180 * M_PI)
                        );
                        cur_id++;
                        try
                          {
                            //std::cout << "Making a point: <" << newPoint << ">\n" << "Spherical coords: [" << shell[i] << ", " << theta << ", " << phi << "]\n";

                            //Modify the find_active_cell_around_point to only search for nearest vertex  and adj. cells, instead of searching all cells in the simulation

                            typename parallel::distributed::Triangulation<dim>::active_cell_iterator it =
                              (GridTools::find_active_cell_around_point<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                              //(GridTools::find_active_cell_around_point_quick<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                            //std::cout << "Point successful!\n";

                            if (it->is_locally_owned())
                              {
                                //Only try to add the point if the cell it is in, is on this processor
                                T new_particle(newPoint, cur_id);
                                world.add_particle(new_particle, std::make_pair(it->level(), it->index()));
                              }
                          }
                        catch (...)
                          {
                            //A point wasn't in an available cell, this might be because it isn't local; we can ignore it

                            //std::cout << "Point failed.\n";
                            //Allow this loss for now
                            //AssertThrow (false, ExcMessage ("Couldn't generate particle (Shell too close to boundary?)."));
                          }
                      }
                  }

    }
  }
}


// explicit instantiations
namespace aspect
{													  -
  namespace Particle
  {
    namespace Generator
    {
    ASPECT_REGISTER_PARTICLE_GENERATOR(RandomUniformGenerator,
                                               "random uniform",
                                               "Generate random uniform distribution of "
                                               "particles over entire simulation domain.")
    }
  }
}
