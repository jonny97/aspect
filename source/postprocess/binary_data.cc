#include <aspect/postprocess/binary_data.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/lac/block_vector.h>

#ifdef DEAL_II_WITH_ZLIB
#  include <zlib.h>
#endif

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    BinaryData<dim>::BinaryData()
    {}

    template <int dim>
    BinaryData<dim>::~BinaryData()
    {}

    template <int dim>
    void BinaryData<dim>::initialize()
    {
      my_id = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      this->update_time();

//           std::ostringstream oss;
//           aspect::oarchive oa (oss);
//           oa << this->get_time();
//
//#ifdef DEAL_II_WITH_ZLIB
//         if (my_id == 0) {
//             uLongf compressed_data_length = compressBound(oss.str().length());
//             std::vector<char *> compressed_data(compressed_data_length);
//             int err = compress2((Bytef *) &compressed_data[0],
//                                 &compressed_data_length,
//                                 (const Bytef *) oss.str().data(),
//                                 oss.str().length(),
//                                 Z_BEST_COMPRESSION);
//             (void) err;
//             Assert (err == Z_OK, ExcInternalError());
//             std::ofstream f((this->get_output_directory() + "time.binary").c_str(), std::ios_base::app);
//             f.write((char *) &compressed_data[0], compressed_data_length);
//         }
//#else
//           AssertThrow (false,
//                   ExcMessage ("You need to have deal.II configured with the 'libz' "
//                               "option to support checkpoint/restart, but deal.II "
//                               "did not detect its presence when you called 'cmake'."));
//#endif
    }

    template< int dim>
    void BinaryData<dim>::update_time()
    {
      attributes.time = this->get_time();
      attributes.time_step = this->get_timestep();
      attributes.old_time_step = this->get_old_timestep();
      attributes.timestep_number = this->get_timestep_number();
    }

    template <int dim>
    std::pair<std::string, std::string> BinaryData<dim>::execute(TableHandler &statistics)
    {
      if (my_id == 0)
        {
          this->update_time();

          //std::ofstream ofs(this->get_output_directory() + "/" + filename_prefix +
          //                  Utilities::int_to_string(this->get_timestep_number(), 5) + ".time");
          std::ofstream ofs(this->get_output_directory() + "fields-" +
                                    Utilities::int_to_string(this->get_timestep_number(), 5) + ".bin");
          boost::archive::binary_oarchive oa(ofs);
          oa << attributes;
          ofs.close();
        }

      std::string fileName = this->get_output_directory() + "/" + filename_prefix + Utilities::int_to_string(this->get_timestep_number(), 5) + ".mesh";
      parallel::distributed::SolutionTransfer<dim, LinearAlgebra::BlockVector> sol_trans(this->get_dof_handler());
      sol_trans.prepare_serialization (this->get_solution());
      this->get_triangulation().save(fileName.c_str());

      /*           std::ofstream output (this->get_output_directory() + "solution-" + Utilities::int_to_string(this->get_timestep_number(), 5) + ".txt");
                 dealii::BlockVector<double> solution(this->get_solution());
                 solution.block_write(output);
      */
      return std::make_pair("Writing binary output to: ", fileName);
    }

    template <int dim>
    void
    BinaryData<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Binary data");
        {
          prm.declare_entry("Binary data file name", "solution-",
                            Patterns::FileName(),
                            "A file name prefix that will be appended to each file that is generated by the "
                            "binary output postprocessor.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    template <int dim>
    void
    BinaryData<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Postprocess");
      {
        prm.enter_subsection("Binary data");
        {
          filename_prefix = prm.get("Binary data file name");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}


namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(BinaryData,
                                  "binary data",
                                  "A Postprocessor that output the velocity solution data per timestep.")
  }
}



