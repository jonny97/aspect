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

#include <aspect/particle/generator.h>
#include <deal.II/grid/grid_tools.h>
#include <sys/time.h>

#include <boost/random.hpp>


namespace aspect
{
  namespace Particle
  {
    namespace Generator
    {
      // Generate random uniform distribution of particles over entire simulation domain
      template <int dim, class T>
      class RandomUniformGenerator : public Interface<dim, T>
      {
        public:
          /**
           * Constructor.
           *
           * @param[in] The MPI communicator for synchronizing particle generation.
           */
          RandomUniformGenerator() {}

          /**
           * Generate a uniformly randomly distributed set of particles in the current triangulation.
           */
          // TODO: fix the particle system so it works even with processors assigned 0 cells
          virtual
          void
          generate_particles(Particle::World<dim, T> &world,
                             const double total_num_particles)
          {
            double      total_volume, local_volume, subdomain_fraction, start_fraction, end_fraction;

            // Calculate the number of particles in this domain as a fraction of total volume
            total_volume = local_volume = 0;
            for (typename parallel::distributed::Triangulation<dim>::active_cell_iterator
                 it=world.get_triangulation()->begin_active();
                 it!=world.get_triangulation()->end(); ++it)
              {
                double cell_volume = it->measure();
                AssertThrow (cell_volume != 0, ExcMessage ("Found cell with zero volume."));

                if (it->is_locally_owned())
                  local_volume += cell_volume;
              }
            // Sum the local volumes over all nodes
            MPI_Allreduce(&local_volume, &total_volume, 1, MPI_DOUBLE, MPI_SUM, world.mpi_comm());

            // Assign this subdomain the appropriate fraction
            subdomain_fraction = local_volume/total_volume;

            // Sum the subdomain fractions so we don't miss particles from rounding and to create unique IDs
            MPI_Scan(&subdomain_fraction, &end_fraction, 1, MPI_DOUBLE, MPI_SUM, world.mpi_comm());
            start_fraction = end_fraction-subdomain_fraction;

            // Calculate start and end IDs so there are no gaps
            // TODO: this can create gaps for certain processor counts because of
            // floating point imprecision, figure out how to fix it
            const unsigned int  start_id = static_cast<unsigned int>(std::ceil(start_fraction*total_num_particles));
            const unsigned int  end_id   = static_cast<unsigned int>(fmin(std::ceil(end_fraction*total_num_particles), total_num_particles));
            const unsigned int  subdomain_particles = end_id - start_id;

            uniform_random_particles_in_subdomain(world, subdomain_particles, start_id);
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
          void uniform_random_particles_in_subdomain (Particle::World<dim, T> &world,
                                                      const unsigned int num_particles,
                                                      const unsigned int start_id)
          {
            unsigned int          i, d, v, num_tries, cur_id;
            double                total_volume, roulette_spin;
            std::map<double, LevelInd>        roulette_wheel;
            const unsigned int n_vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
            Point<dim>            pt, max_bounds, min_bounds;
            LevelInd              select_cell;

            // Create the roulette wheel based on volumes of local cells
            total_volume = 0;
            for (typename parallel::distributed::Triangulation<dim>::active_cell_iterator
                 it=world.get_triangulation()->begin_active(); it!=world.get_triangulation()->end(); ++it)
              {
                if (it->is_locally_owned())
                  {
                    // Assign an index to each active cell for selection purposes
                    total_volume += it->measure();
                    // Save the cell index and level for later access
                    roulette_wheel.insert(std::make_pair(total_volume, std::make_pair(it->level(), it->index())));
                  }
              }

            // Pick cells and assign particles at random points inside them
            cur_id = start_id;
            for (i=0; i<num_particles; ++i)
              {
                // Select a cell based on relative volume
                roulette_spin = total_volume*uniform_distribution_01(random_number_generator);
                select_cell = roulette_wheel.lower_bound(roulette_spin)->second;

                const typename parallel::distributed::Triangulation<dim>::active_cell_iterator
                it (world.get_triangulation(), select_cell.first, select_cell.second);

                // Get the bounds of the cell defined by the vertices
                for (d=0; d<dim; ++d)
                  {
                    min_bounds[d] = INFINITY;
                    max_bounds[d] = -INFINITY;
                  }
                for (v=0; v<n_vertices_per_cell; ++v)
                  {
                    pt = it->vertex(v);
                    for (d=0; d<dim; ++d)
                      {
                        min_bounds[d] = fmin(pt[d], min_bounds[d]);
                        max_bounds[d] = fmax(pt[d], max_bounds[d]);
                      }
                  }

                // Generate random points in these bounds until one is within the cell
                num_tries = 0;
                while (num_tries < 100)
                  {
                    for (d=0; d<dim; ++d)
                      {
                        pt[d] = uniform_distribution_01(random_number_generator) *
                                (max_bounds[d]-min_bounds[d]) + min_bounds[d];
                      }
                    try
                      {
                        if (it->point_inside(pt)) break;
                      }
                    catch (...)
                      {
                        // Debugging output, remove when Q4 mapping 3D sphere problem is resolved
                        //std::cerr << "Pt and cell " << pt << " " << select_cell.first << " " << select_cell.second << std::endl;
                        //for (int z=0;z<8;++z) std::cerr << "V" << z <<": " << it->vertex(z) << ", ";
                        //std::cerr << std::endl;
                        //***** MPI_Abort(communicator, 1);
                      }
                    num_tries++;
                  }
                AssertThrow (num_tries < 100, ExcMessage ("Couldn't generate particle (unusual cell shape?)."));

                // Add the generated particle to the set
                T new_particle(pt, cur_id);
                world.add_particle(new_particle, select_cell);

                cur_id++;
              }
          }

        private:
          /**
           * Random number generator and an object that describes a
           * uniform distribution on the interval [0,1]. These
           * will be used to generate particle locations at random.
           */
          boost::mt19937            random_number_generator;
          boost::uniform_01<double> uniform_distribution_01;
      };

      // Generate uniform distribution of particles radially within given constraints (angle and radius)
      template <int dim, class T>
      class UniformRadialGenerator : public InterfaceLimits<dim, T>
      {
        public:
          /**
           * Constructor.
           *
           * @param[in] The MPI communicator for synchronizing particle generation.
           */
          UniformRadialGenerator() {}

          virtual
          void
          generate_particles(Particle::World<dim, T> &world,
                             const double total_num_particles)
          {
            return;
          };

          /**
           * Generate a uniformly randomly distributed set of particles in the current triangulation.
           *
           * @param[in] radialLevels The amount of radial levels between the min and max shell. If set to 1, only min radius is used.
           * @param[in] limits The limits within which to generate points; format [shellCount, radiusMin, radiusMax, thetaMin, thetaMax, phiMin, phiMax].
           */
          // TODO: fix the particle system so it works even with processors assigned 0 cells
          virtual
          void
          generate_particles(Particle::World<dim, T> &world,
                             const double total_num_particles,
                             double *limits)
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
           * @param[in] world The particle world the particles will exist in
           * @param[in] start_id The starting ID to assign to generated particles
           * @param[in] shell An array holding the shell at which to generate particles.
           * @param[in] particlesPerRadius An array with the amount of particles per shell.
           * @param[in] limits The limits within which to generate points; format [radiusMin, radiusMax, thetaMin, thetaMax, phiMin, phiMax].
           */
          void uniform_radial_particles_in_subdomain (Particle::World<dim, T> &world,
                                                      const unsigned int start_id,
                                                      double *shell,
                                                      int *particlesPerRadius,
                                                      double *limits,
                                                      int shellCount)
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
                                  //(GridTools::find_active_cell_around_point<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                                  (GridTools::find_active_cell_around_point_quick<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
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
                              //(GridTools::find_active_cell_around_point<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                              (GridTools::find_active_cell_around_point_quick<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
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
          };
      };



      // Generate uniform distribution of particles radially within given constraints (x, y, z)
      template <int dim, class T>
      class UniformBoxGenerator : public InterfaceLimits<dim, T>
      {
        public:
          /**
           * Constructor.
           *
           * @param[in] The MPI communicator for synchronizing particle generation.
           */
          UniformBoxGenerator() {}

          virtual
          void
          generate_particles(Particle::World<dim, T> &world,
                             const double total_num_particles)
          {
            return;
          };

          /**
           * Generate a uniformly randomly distributed set of particles in the current triangulation.
           *
           * @param[in] radialLevels The amount of radial levels between the min and max shell. If set to 1, only min radius is used.
           * @param[in] limits The limits within which to generate points; format [shellCount, radiusMin, radiusMax, thetaMin, thetaMax, phiMin, phiMax].
           */
          // TODO: fix the particle system so it works even with processors assigned 0 cells
          virtual
          void
          generate_particles(Particle::World<dim, T> &world,
                             const double total_num_particles,
                             double *limits)
          {
            //double      subdomain_fraction, start_fraction, end_fraction;
            int size, rank, localParticleCount, startID, endID;
            double *shellArray, shellSeperation;

            //std::cout << " limits[2]: " << limits[2] << "\nlimits[1]: " << limits[1] << "\n";
            //std::cout << " limits[4]: " << limits[4] << "\nlimits[3]: " << limits[3] << "\n";

            //Does wall clock keeping in seconds
            struct timeval tp;
            struct timezone tzp;
            gettimeofday (&tp, &tzp);
            double wall_seconds = ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);

            //Does CPU clock keeping in seconds
            double cpu_seconds = (double)clock() / CLOCKS_PER_SEC;

            localParticleCount = total_num_particles;

            //Since we generate each particle on each processor, the above is irrelevant
            startID = 0;

            //std::cout << "Test Pre-subdomain\n";

            uniform_radial_particles_in_subdomain(world, startID, limits, localParticleCount);


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
           * @param[in] world The particle world the particles will exist in
           * @param[in] start_id The starting ID to assign to generated particles
           * @param[in] shell An array holding the shell at which to generate particles.
           * @param[in] particlesPerRadius An array with the amount of particles per shell.
           * @param[in] limits The limits within which to generate points; format [radiusMin, radiusMax, thetaMin, thetaMax, phiMin, phiMax].
           */
          void uniform_radial_particles_in_subdomain (Particle::World<dim, T> &world,
                                                      const unsigned int start_id,
                                                      double *limits,
                                                      int localParticleCount)
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
                                  //(GridTools::find_active_cell_around_point<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
                                  (GridTools::find_active_cell_around_point_quick<> (*(world.get_mapping()), *(world.get_triangulation()), newPoint)).first;
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


      template <int dim, class T>
      Interface<dim,T> *
      create_generator_object (const std::string &generator_type)
      {
        if (generator_type == "random_uniform")
          return new RandomUniformGenerator<dim,T>();
        else if (generator_type == "radial_uniform")
          return new UniformRadialGenerator<dim,T>();
        else if (generator_type == "box_uniform")
          return new UniformBoxGenerator<dim,T>();
        else
          Assert (false, ExcNotImplemented());

        return 0;
      }


      std::string
      generator_object_names ()
      {
        return ("random_uniform|radial_uniform|box_uniform");
      }


      // explicit instantiations
      template
      Interface<2,Particle::BaseParticle<2> > *
      create_generator_object (const std::string &generator_type);
      template
      Interface<3,Particle::BaseParticle<3> > *
      create_generator_object (const std::string &generator_type);
    }
  }
}
