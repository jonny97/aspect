//-------------------------------------------------------------
//    $Id: simulator.h 232 2011-10-19 13:30:15Z bangerth $
//
//    Copyright (C) 2011 by the authors of the ASPECT code
//
//-------------------------------------------------------------
#ifndef __aspect__equation_data_h
#define __aspect__equation_data_h


#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>



//TODO: should move into namespace aspect
namespace EquationData
{
  using namespace dealii;

  extern const double year_in_seconds;


  template <int dim>
  double adiabatic_pressure (const Point<dim> &p);

  template <int dim>
  double adiabatic_temperature (const Point<dim> &p);


  namespace MaterialModel
  {
    template <int dim>
    double eta (const double temperature, const double pressure, const Point<dim> &position);

    template <int dim>
    double real_viscosity (const double                 temperature,
                           const double                  pressure,
                           const Point<dim> &position,
                           const SymmetricTensor<2, dim> &strain_rate);
    template <int dim>
    double density (const double temperature,
                    const double pressure,
                    const Point<dim> &position);
  }
}


#endif
