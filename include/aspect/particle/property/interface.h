/*
 Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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

#ifndef __aspect__particle_property_interface_h
#define __aspect__particle_property_interface_h

#include <aspect/particle/base_particle.h>
#include <aspect/particle/definitions.h>

#include <aspect/simulator_access.h>
#include <aspect/plugins.h>
#include <deal.II/base/std_cxx1x/shared_ptr.h>


namespace aspect
{
  namespace Particle
  {
    namespace Property
    {

      /**
        * Interface provides an example of how to extend the BaseParticle
        * class to include related particle data. This allows users to attach
        * scalars/vectors/tensors/etc to particles and ensure they are
        * transmitted correctly over MPI and written to output files.
        */
       template <int dim>
       class Interface
       {
         public:

           /**
            * Initialization function. This function is called once at the
            * beginning of the program after parse_parameters is run.
            */
           virtual
           void
           initialize ();

           /**
            * Initialization function. This function is called once at the
            * creation of every particle to set up it's properties.
            */
           virtual
           void
           initialize_particle (std::vector<double> &/*data*/,
                                const Point<dim> &/*position*/,
                                const Vector<double> &/*solution*/,
                                const std::vector<Tensor<1,dim> > &/*gradients*/);

           /**
            * Update function. This function is called every timestep for
            * every particle to update it's properties. It is obvious that
            * this function is called a lot, so its code should be efficient.
            */
           virtual
           void
           update_particle (unsigned int data_position,
                            std::vector<double> &/*particle_properties*/,
                            const Point<dim> &/*position*/,
                            const Vector<double> &/*solution*/,
                            const std::vector<Tensor<1,dim> > &/*gradients*/);

           /**
            * Returns a bool, which is false in the default implementation,
            * telling the property manager that no update is needed. Every
            * plugin that implements this function should return true. This
            * saves considerable computation time in cases, when no plugin needs
            * to update tracer properties over time.
            */
           virtual
           bool
           need_update ();

           virtual unsigned int data_len() const;

           /**
            * Set up the MPI data type information for the Interface type
            *
            * @param [in,out] data_info Vector to append MPIDataInfo objects to
            */
           virtual
           void add_mpi_types(std::vector<MPIDataInfo> &data_info) const = 0;


           /**
            * Declare the parameters this class takes through input files.
            * Derived classes should overload this function if they actually do
            * take parameters; this class declares a fall-back function that
            * does nothing, so that property classes that do not take any
            * parameters do not have to do anything at all.
            *
            * This function is static (and needs to be static in derived
            * classes) so that it can be called without creating actual objects
            * (because declaring parameters happens before we read the input
            * file and thus at a time when we don't even know yet which
            * property objects we need).
            */
           static
           void
           declare_parameters (ParameterHandler &prm);

           /**
            * Read the parameters this class declares from the parameter file.
            * The default implementation in this class does nothing, so that
            * derived classes that do not need any parameters do not need to
            * implement it.
            */
           virtual
           void
           parse_parameters (ParameterHandler &prm);
       };



    /**
     * Manager class of properties - This class sets the data of the particles
     * and updates it over time if requested by the user selected properties
     */
    template <int dim>
    class Manager : public SimulatorAccess<dim>
    {
      private:
        /**
         * A list of property objects that have been requested in the
         * parameter file.
         */
        std::list<std_cxx1x::shared_ptr<Interface<dim> > > property_list;

        /**
         * A map between names of properties and their data component.
         */
        std::map<std::string,unsigned int> property_component_map;

        /**
         * The number of doubles needed to represent a typical tracer
         */
        unsigned int data_len;


      public:
        /**
         * Empty constructor for Manager
         */
        Manager ();

        /**
         * Destructor for Manager
         */
        virtual
        ~Manager ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program after parse_parameters is run.
         */
        virtual
        void
        initialize ();

        /**
         * Initialization function for particle properties. This function is
         * called once at the creation of a particle
         */
        virtual
        void
        initialize_particle (BaseParticle<dim> &particle,
                             const Vector<double> &solution,
                             const std::vector<Tensor<1,dim> > &gradients);

        /**
         * Update function for particle properties. This function is
         * called once every timestep for every particle
         */
        virtual
        void
        update_particle (BaseParticle<dim> &particle,
                         const Vector<double> &solution,
                         const std::vector<Tensor<1,dim> > &gradients);

        /**
         * Returns a bool, which is false if no selected plugin needs to
         * update tracer properties over time. This saves considerable
         * computation time in cases, when no plugin needs to update tracer
         * properties over time, because the solution does not need to be
         * evaluated at tracer positions in this case.
         */
        virtual
        bool
        need_update ();

        /**
         * Get the number of doubles required to represent this particle's
         * properties for communication.
         *
         * @return Number of doubles required to represent this particle
         */
        unsigned int
        get_data_len () const;

        /**
         * Add the MPI data description for this particle type to the vector.
         *
         * @param[in,out] data_info Vector to which MPI data description is
         * appended.
         */
        void
        add_mpi_types (std::vector<aspect::Particle::MPIDataInfo> &data_info) const;

        /**
         * Initialize the property map from the given vector of MPIDataInfo.
         * This map will be used to query the position of a particular property
         * in the property vector of each particle.
         */
        void
        initialize_property_map (const std::vector<aspect::Particle::MPIDataInfo> &data_info);

        /**
         * Get the position of the property specified by name in the property
         * vector of the particles.
         */
        unsigned int
        get_property_component_by_name(const std::string &name) const;

        /**
         * A function that is used to register visualization postprocessor
         * objects in such a way that the Manager can deal with all of them
         * without having to know them by name. This allows the files in which
         * individual properties are implemented to register these
         * properties, rather than also having to modify the Manager class
         * by adding the new properties class.
         *
         * @param name The name under which this particle property
         * is to be called in parameter files.
         * @param description A text description of what this particle property
         *  does and that will be listed in the documentation of the
         * parameter file.
         * @param declare_parameters_function A pointer to a function that
         * declares the parameters for this property.
         * @param factory_function A pointer to a function that creates such a
         * property object and returns a pointer to it.
         */
        static
        void
        register_particle_property (const std::string &name,
                                              const std::string &description,
                                              void (*declare_parameters_function) (ParameterHandler &),
                                              Property::Interface<dim> *(*factory_function) ());


        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

    };


    /**
     * Given a class name, a name, and a description for the parameter file for
     * a tracer property, register it with the aspect::Particle:: class.
     *
     * @ingroup Particle
     */
    #define ASPECT_REGISTER_PARTICLE_PROPERTY(classname,name,description) \
    template class classname<2>; \
    template class classname<3>; \
    namespace ASPECT_REGISTER_PARTICLE_PROPERTY_ ## classname \
    { \
      aspect::internal::Plugins::RegisterHelper<aspect::Particle::Property::Interface<2>,classname<2> > \
      dummy_ ## classname ## _2d (&aspect::Particle::Property::Manager<2>::register_particle_property, \
                                  name, description); \
      aspect::internal::Plugins::RegisterHelper<aspect::Particle::Property::Interface<3>,classname<3> > \
      dummy_ ## classname ## _3d (&aspect::Particle::Property::Manager<3>::register_particle_property, \
                                  name, description); \
    }

    }
  }
}

#endif
