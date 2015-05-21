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

#include <aspect/particle/generator/uniform_box.h>

#include <boost/random.hpp>


namespace aspect
{
  namespace Particle
  {
    namespace Generator
    {
      // Generate random uniform distribution of particles over entire simulation domain

          /**
           * Constructor.
           *
           * @param[in] The MPI communicator for synchronizing particle generation.
           */
        template <int dim>
        UniformBox<dim>::UniformBox() {}

          /**
           * TODO: Update comments
           * Generate a uniformly randomly distributed set of particles in the current triangulation.
           */
          // TODO: fix the particle system so it works even with processors assigned 0 cells
        template <int dim>
        void
        UniformBox<dim>::generate_particles(Particle::World<dim> &world,
                                                        const double total_num_particles)
          {
            //double      subdomain_fraction, start_fraction, end_fraction;
            int size, rank, shellCount, shellRemainder, shellStart, shellEnd, *ppr, localParticleCount, startID, endID, radiusTotal;
            double *shellArray, shellSeperation;
            int radialLevels = limits[0];

            //Does wall clock keeping in seconds
            struct timeval tp;
            struct timezone tzp;
            gettimeofday (&tp, &tzp);
            double wall_seconds = ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);

            //Does CPU clock keeping in seconds
            double cpu_seconds = (double)clock() / CLOCKS_PER_SEC;


            shellCount = radialLevels;


            /* //Work by generating the particles on every processor
               //Afterwards, generate a bounding box for every cell

            // Find how many nodes there are, so that we may distribute shell, and the ID of this one.
            size = MPI_Comm_size(world.mpi_comm(), &size);
            rank = MPI_Comm_rank(world.mpi_comm(), &rank);
            if (size == 0)
              size = 1;
            shellCount = radialLevels / size;
            shellRemainder = radialLevels % size;

            // If the ID is less than the remainder, this node gets one extra shell to deal with...
            if (rank < shellRemainder)
            {
              shellCount++;
            }

            // Get the local starting shell
            MPI_Scan(&shellCount, &shellEnd, 1, MPI_INT, MPI_SUM, world.mpi_comm());
            shellStart = shellEnd - shellCount;*/

            // Create the array of shell to deal with
            shellArray = new double[shellCount];
            shellSeperation = (limits[2] - limits[1]) / radialLevels;
            shellStart = 0;
            ppr = new int[shellCount];
            localParticleCount = 0;
            radiusTotal = 0;

            for (int i = 0; i < shellCount; ++i)
              {
                // Calculate radius of each shell
                radiusTotal += shellArray[i] = limits[1] + (shellSeperation * (shellStart + i));
              }

            for (int i = 0; i < shellCount; ++i)
              {
                // Calculate amount of particles per shell.
                // Number of particles depend on the portion of the radius that this shell is in (i.e., more radius = more particles)
                ppr[i] = round(total_num_particles * shellArray[i] / radiusTotal);
                //cout << "Shell " << i << ": " << ppr[i] << " particles.\n";
                localParticleCount += ppr[i];
              }

            //To give each particle a unique ID, add IDs throughout all nodes
            //MPI_Scan(&localParticleCount, &endID, 1, MPI_INT, MPI_SUM, world.mpi_comm());
            //startID = endID - localParticleCount;

            //Since we generate each particle on each processor, the above is irrelevant
            startID = 0;

            uniform_radial_particles_in_subdomain(world, startID, shellArray, ppr, limits, shellCount);

            delete ppr;
            delete shellArray;

            gettimeofday (&tp, &tzp);
            wall_seconds = ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6) - wall_seconds;
            cpu_seconds = ((double)clock() / CLOCKS_PER_SEC) - cpu_seconds;

            std::cout << "Time taken (wall): " << wall_seconds << "\n";
            std::cout << "Time taken (cpu) : " << cpu_seconds  << "\n";

          };


          /**
           * Generate a set of particles uniformly randomly distributed within the
           * specified triangulation. This is done using "roulette wheel" style
           * selection weighted by cell volume. We do cell-by-cell assignment of
           * particles because the decomposition of the mesh may result in a highly
           * non-rectangular local mesh which makes uniform particle distribution difficult.
           *
           * @param [in] world The particle world the particles will exist in
           * @param [in] num_particles The number of particles to generate in this subdomain
           * @param [in] start_id The starting ID to assign to generated particles
           */
        template <int dim>
          void
          UniformBox<dim>::uniform_random_particles_in_subdomain (Particle::World<dim> &world,
                                                      const unsigned int num_particles,
                                                      const unsigned int start_id)
												         {
            unsigned int cur_id;
            cur_id = start_id;
            //std::cout << "Pre-Dim\n";
            if (dim == 3)
              {
                double xDiff = limits[2] - limits[1];
                double yDiff = limits[4] - limits[3];
                double zDiff = limits[6] - limits[5];
                double totalDiff = xDiff + yDiff + zDiff;
                int xParticles, yParticles, zParticles;

                ///Amount of particles is the total amount of particles, divided by length of each axis
                xParticles = round(localParticleCount * xDiff / totalDiff);
                yParticles = round(localParticleCount * yDiff / totalDiff);
                zParticles = round(localParticleCount * zDiff / totalDiff);

                //xParticles = floor(pow(localParticleCount, (1/3)) * xDiff / totalDiff);
                //yParticles = floor(pow(localParticleCount, (1/3)) * yDiff / totalDiff);
                //zParticles = floor(pow(localParticleCount, (1/3)) * zDiff / totalDiff);

                double xSeperation, ySeperation, zSeperation;
                xSeperation = xDiff / xParticles;
                ySeperation = yDiff / yParticles;
                zSeperation = zDiff / zParticles;

                for (double i = limits[1]; i < limits[2]; i+= xSeperation)
                  {
                    for (double j = limits[3]; j < limits[4]; j += ySeperation)
                      {
                        for (double k = limits[5]; k < limits[6]; k += zSeperation)
                          {
                            Point<dim> newPoint (i,j,k);
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
                                    BaseParticle<dim> new_particle(newPoint, cur_id);
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
            else if (dim == 2)
              {
                //std::cout << "Dim-2 Early\n";
                double xDiff = limits[2] - limits[1];
                double yDiff = limits[4] - limits[3];
                double totalDiff = xDiff + yDiff;
                int xParticles, yParticles;

                ///Amount of particles is the total amount of particles, divided by length of each axis
                xParticles = round(sqrt(localParticleCount));
                yParticles = round(localParticleCount / xParticles);

                //std::cout << "limits[2]: " << limits[2] << "\nlimits[1]: " << limits[1] << "\n";
                //std::cout << "limits[4]: " << limits[4] << "\nlimits[3]: " << limits[3] << "\n";

                //std::cout << "xParts: " << xParticles << "\nxDiff: " << xDiff << "\n";
                //std::cout << "yParts: " << yParticles << "\nyDiff: " << yDiff << "\n";


                //return;

                double xSeperation, ySeperation;
                xSeperation = xDiff / xParticles;
                ySeperation = yDiff / yParticles;

                //std::cout << "xSeperation: " << xSeperation << "\n";
                //std::cout << "ySeperation: " << ySeperation << "\n";

                //std::cout << "Dim-2 Pre-Loop\n";
                for (double i = limits[1]; i < limits[2]; i+= xSeperation)
                  {
                    //std::cout << "Dim-2 Loop I:" << i << " | " << limits[2] << "\n";
                    for (double j = limits[3]; j < limits[4]; j += ySeperation)
                      {
                        //std::cout << "  Dim-2 Loop J:" << j << " | " << limits[4] << "\n";
                        Point<dim> newPoint (i,j);
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

                            std::cout << "Point failed.\n";
                            //Allow this loss for now
                            AssertThrow (false, ExcMessage ("Couldn't generate particle (Shell too close to boundary?)."));
                          }
                      }
                  }
              }
          };
      };
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Generator
    {
    ASPECT_REGISTER_PARTICLE_GENERATOR(UniformBox,
                                               "uniform box",
                                               "Generate a uniform distribution of particles"
                                               " over a rectangular domain in or or 3D.")
    }
  }
}
