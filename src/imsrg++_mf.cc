/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
///                                                  ____                                         ///
///        _________________           _____________/   /\               _________________        ///
///       /____/_____/_____/|         /____/_____/ /___/  \             /____/_____/_____/|       ///
///      /____/_____/__G_ /||        /____/_____/|/   /\  /\           /____/_____/____ /||       ///
///     /____/_____/__+__/|||       /____/_____/|/ G /  \/  \         /____/_____/_____/|||       ///
///    |     |     |     ||||      |     |     |/___/   /\  /\       |     |     |     ||||       ///
///    |  I  |  M  |     ||/|      |  I  |  M  /   /\  /  \/  \      |  I  |  M  |     ||/|       ///
///    |_____|_____|_____|/||      |_____|____/ + /  \/   /\  /      |_____|_____|_____|/||       ///
///    |     |     |     ||||      |     |   /___/   /\  /  \/       |     |     |     ||||       ///
///    |  S  |  R  |     ||/|      |  S  |   \   \  /  \/   /        |  S  |  R  |  G  ||/|       ///
///    |_____|_____|_____|/||      |_____|____\ __\/   /\  /         |_____|_____|_____|/||       ///
///    |     |     |     ||||      |     |     \   \  /  \/          |     |     |     ||||       ///
///    |     |  +  |     ||/       |     |  +  |\ __\/   /           |     |  +  |  +  ||/        ///
///    |_____|_____|_____|/        |_____|_____|/\   \  /            |_____|_____|_____|/         ///
///                                               \___\/                                          ///
///                                                                                               ///
///           imsrg++ : Interface for performing standard IMSRG calculations.                     ///
///                     Usage is imsrg++  option1=value1 option2=value2 ...                       ///
///                     To get a list of options, type imsrg++ help                               ///
///                                                                                               ///
///                                                      - Ragnar Stroberg 2016                   ///
///                                                                                               ///
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
//    imsrg++.cc, part of  imsrg++
//    Copyright (C) 2018  Ragnar Stroberg
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License along
//    with this program; if not, write to the Free Software Foundation, Inc.,
//    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
///////////////////////////////////////////////////////////////////////////////////

#include <mpi.h>

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <omp.h>
#include "Commutator.hh"
#include "IMSRG.hh"
#include "Operator.hh"
#include "Parameters.hh"
#include "PhysicalConstants.hh"
#include "imsrg_util.hh"
#include "version.hh"

struct OpFromFile {
   std::string file2name,file3name,opname;
   int j,p,t,r; // J rank, parity, dTz, particle rank
};

int main(int argc, char** argv)
{
  // Default parameters, and everything passed by command line args.
  std::cout << "######  imsrg++ build version: " << version::BuildVersion() << std::endl;

  int world_size = 1;
  int my_rank = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  Parameters parameters(argc,argv);
  if (parameters.help_mode) return 0;

  std::string inputtbme = parameters.s("2bme");
  std::string inputtbmeNO2B = parameters.s("2bmeNO2B");
  std::string input3bme = parameters.s("3bme");
  std::string input3bme_type = parameters.s("3bme_type");
  std::string no2b_precision = parameters.s("no2b_precision");
  std::string reference = parameters.s("reference");
  std::string valence_space = parameters.s("valence_space");
  std::string custom_valence_space = parameters.s("custom_valence_space");
  std::string basis = parameters.s("basis");
  std::string method = parameters.s("method");
  std::string flowfile = parameters.s("flowfile") + "_Rank" + std::to_string(my_rank);
  std::string intfile = parameters.s("intfile") + "_Rank" + std::to_string(my_rank);
  std::string core_generator = parameters.s("core_generator");
  std::string valence_generator = parameters.s("valence_generator");
  std::string fmt2 = parameters.s("fmt2");
  std::string fmt3 = parameters.s("fmt3");
  std::string input_op_fmt = parameters.s("input_op_fmt");
  std::string denominator_delta_orbit = parameters.s("denominator_delta_orbit");
  std::string LECs = parameters.s("LECs");
  std::string scratch = parameters.s("scratch");
  std::string valence_file_format = parameters.s("valence_file_format");
  std::string occ_file = parameters.s("occ_file");
  std::string physical_system = parameters.s("physical_system");
  std::string denominator_partitioning = parameters.s("denominator_partitioning");
  std::string NAT_order = parameters.s("NAT_order");

  bool use_brueckner_bch = parameters.s("use_brueckner_bch") == "true";
  bool nucleon_mass_correction = parameters.s("nucleon_mass_correction") == "true";
  bool relativistic_correction = parameters.s("relativistic_correction") == "true";
  bool IMSRG3 = parameters.s("IMSRG3") == "true";
  bool imsrg3_n7 = parameters.s("imsrg3_n7") == "true";
  bool reduced_232_impl = parameters.s("reduced_232_impl") == "true";
  bool imsrg3_mp4 = parameters.s("imsrg3_mp4") == "true";
  bool imsrg3_at_end = parameters.s("imsrg3_at_end") == "true";
  bool imsrg3_no_qqq = parameters.s("imsrg3_no_qqq") == "true";
  bool write_omega = parameters.s("write_omega") == "true";
  bool freeze_occupations = parameters.s("freeze_occupations")=="true";
  bool discard_no2b_from_3n = parameters.s("discard_no2b_from_3n")=="true";
  bool hunter_gatherer = parameters.s("hunter_gatherer") == "true";
  bool goose_tank = parameters.s("goose_tank") == "true";
  bool discard_residual_input3N = parameters.s("discard_residual_input3N")=="true";
  bool use_NAT_occupations = (parameters.s("use_NAT_occupations")=="true") ? true : false;
  bool order_NAT_by_energy = (parameters.s("order_NAT_by_energy")=="true") ? true : false;
  bool store_3bme_pn = (parameters.s("store_3bme_pn")=="true");
  bool only_2b_eta = (parameters.s("only_2b_eta")=="true");
  bool only_2b_omega = (parameters.s("only_2b_omega")=="true");
  bool only_2b_omega_at_end = (parameters.s("only_2b_omega_at_end")=="true");
  bool perturbative_triples = (parameters.s("perturbative_triples")=="true");
  bool brueckner_restart = false;
  bool write_HO_ops = parameters.s("write_HO_ops") == "true";  // added by Antoine Belley
  bool write_HF_ops = parameters.s("write_HF_ops") == "true";  // added by Antoine Belley
  bool use_HF_reference_in_NAT = parameters.s("use_HF_reference_in_NAT") == "true";
  bool use_HF_valence_in_NAT = parameters.s("use_HF_valence_in_NAT") == "true";

  int eMax = parameters.i("emax");
  int lmax = parameters.i("lmax"); // so far I only use this with atomic systems.
  int E3max = parameters.i("e3max");
  int lmax3 = parameters.i("lmax3");
  int targetMass = parameters.i("A");
  int nsteps = parameters.i("nsteps");
  int file2e1max = parameters.i("file2e1max");
  int file2e2max = parameters.i("file2e2max");
  int file2lmax = parameters.i("file2lmax");
  int file3e1max = parameters.i("file3e1max");
  int file3e2max = parameters.i("file3e2max");
  int file3e3max = parameters.i("file3e3max");
  int atomicZ = parameters.i("atomicZ");
  int emax_unocc = parameters.i("emax_unocc");
  int eMax_imsrg = parameters.i("emax_imsrg");
  int e2Max_imsrg = parameters.i("e2max_imsrg");
  int e3Max_imsrg = parameters.i("e3max_imsrg");
  int eMax_3body_imsrg = parameters.i("emax_3body_imsrg");
  int imsrg3_commutator_depth = parameters.i("imsrg3_commutator_depth");
//  if ( not ( eMax_imsrg==-1 and e2Max_imsrg==-1 and e3Max_imsrg==-1 ) )
//  {
//    if ( eMax_imsrg==-1 ) eMax_imsrg = eMax;
//    if ( e2Max_imsrg==-1 ) e2Max_imsrg = 2*eMax_imsrg;
//    if ( e3Max_imsrg==-1 ) e3Max_imsrg = std::min( E3max, 3*eMax_imsrg);
//  }
////  if (e2Max_imsrg==-1 and eMax_imsrg != -1) e2Max_imsrg = 2*eMax_imsrg;
////  if (e3Max_imsrg==-1 and eMax_imsrg != -1) e3Max_imsrg = std::min(E3max, 3*eMax_imsrg);

  double hw = parameters.d("hw");
  double smax = parameters.d("smax");
  double ode_tolerance = parameters.d("ode_tolerance");
  double dsmax = parameters.d("dsmax");
  double ds_0 = parameters.d("ds_0");
  double domega = parameters.d("domega");
  double omega_norm_max = parameters.d("omega_norm_max");
  double denominator_delta = parameters.d("denominator_delta");
  double BetaCM = parameters.d("BetaCM");
  double hwBetaCM = parameters.d("hwBetaCM");
  double eta_criterion = parameters.d("eta_criterion");
  double hw_trap = parameters.d("hw_trap");
  double dE3max = parameters.d("dE3max");
  double OccNat3Cut = parameters.d("OccNat3Cut");
  double threebody_threshold = parameters.d("threebody_threshold");

  std::vector<std::string> opnames = parameters.v("Operators");
  std::vector<std::string> opsfromfile = parameters.v("OperatorsFromFile");
  std::vector<std::string> opnamesPT1 = parameters.v("OperatorsPT1");
  std::vector<std::string> opnamesRPA = parameters.v("OperatorsRPA");
  std::vector<std::string> opnamesTDA = parameters.v("OperatorsTDA");

  std::vector<Operator> ops;
  std::vector<std::string> spwf = parameters.v("SPWF");

  using PhysConst::PROTON_RCH2;
  using PhysConst::NEUTRON_RCH2;
  using PhysConst::DARWIN_FOLDY;


  // test 2bme file
  if (inputtbme != "none" and fmt2.find("oakridge")==std::string::npos and fmt2 != "schematic" )
  {
    if( not std::ifstream(inputtbme).good() )
    {
      std::cout << "trouble reading " << inputtbme << "  fmt2 = " << fmt2 << "   exiting. " << std::endl;
      return 1;
    }
  }
  // test 3bme file
  if (input3bme != "none")
  {
    if( not std::ifstream(input3bme).good() )
    {
      std::cout << "trouble reading " << input3bme << " exiting. " << std::endl;
      return 1;
    }
  }

  // unpack the awkward input format for reading an operator from file, and put it into a struct.
  // the format should look like OpName^j_t_p_r^/path/to/2bfile^/path/to/3bfile  if particle rank of Op is 2-body, then 3bfile is not needed.
  std::vector< OpFromFile> opsfromfile_unpacked;
  // If we're reading in other operators, make sure those are ok too
  for (auto& tag : opsfromfile)
  {
     std::istringstream ss(tag);
     std::string opname,qnumbers,f2name,f3name="";

     OpFromFile opff;
  
     getline(ss,opname,'^');
     getline(ss,qnumbers,'^');
     getline(ss,f2name,'^');
     if ( not ss.eof() )  getline(ss,f3name,'^');
     opff.opname = opname;
     opff.file2name = f2name;
     opff.file3name = f3name;

      ss.str(qnumbers);
      ss.clear();
      std::string tmp;
      getline(ss,tmp,'_');
      std::istringstream(tmp) >> opff.j;
      getline(ss,tmp,'_');
      std::istringstream(tmp) >> opff.t;
      getline(ss,tmp,'_');
      std::istringstream(tmp) >> opff.p;
      getline(ss,tmp,'_');
      std::istringstream(tmp) >> opff.r;
      
      std::cout << "Parsed tag. opname = " << opff.opname << "  " << opff.j << " " << opff.t << " " << opff.p << " " << opff.r << "   file2 = " << opff.file2name   << "    file3 = " << opff.file3name << std::endl;

      // now make sure the files exist before we add them to the list.

//     if( not std::ifstream(f2name).good() )
if (opff.file2name != "") {
     if( not std::ifstream(opff.file2name).good() )
     {
//       std::cout << "trouble reading " << f2name << " exiting. " << std::endl;
       std::cout << "trouble reading " << opff.file2name << " exiting. " << std::endl;
       return 1;
     }
}

     if ( opff.file3name != "") // is there a 3-body file too?
     {
//       getline(ss,f3name,'^');
//       if( not std::ifstream(f3name).good() )
       if( not std::ifstream(opff.file3name).good() )
       {
         std::cout << "trouble reading " << opff.file3name << " exiting. " << std::endl;
//         std::cout << "trouble reading " << f3name << " exiting. " << std::endl;
         return 1;
       }
     }
     // if the files look good, then add it to the list
     opsfromfile_unpacked.push_back( opff );
  }



  ReadWrite rw;
  rw.SetLECs_preset(LECs);
  rw.SetScratchDir(scratch);
  rw.Set3NFormat( fmt3 );



  // deal with some short-hand method names
  if (method == "NSmagnus") // "No split" magnus
  {
    omega_norm_max=50000;
    method = "magnus";
  }
  if (method.find("brueckner") != std::string::npos)
  {
    if (method=="brueckner2") brueckner_restart=true;
    if (method=="brueckner1step")
    {
       nsteps = 1;
       core_generator = valence_generator;
    }
    use_brueckner_bch = true;
    omega_norm_max=500;
    method = "magnus";
  }



  // Test whether the scratch directory exists and we can write to it.
  // This is necessary because otherwise you get garbage for transformed operators and it's
  // not obvious what went wrong.
  if ( ((method == "magnus") || (method == "magnus_backoff")) and  ( (opnames.size() + opsfromfile.size()) > 0 )  )
  {
    if ( scratch=="/dev/null" or scratch=="/dev/null/")
    {
      std::cout << "ERROR!!! using Magnus with scratch = " << scratch << " but you're also trying to transform some operators. Dying now. " << std::endl;
      exit(EXIT_FAILURE);
    }
    else if ( scratch != "" )
    {
      std::string testfilename = scratch + "/_this_is_a_test_delete_me_" + std::to_string(my_rank);
      std::ofstream testout(testfilename);
      testout << "PASSED" << std::endl;
      testout.close();

      // now read it back.
      std::ifstream testin(testfilename);
      std::string checkpassed;
      testin >> checkpassed;
      if ( (checkpassed != "PASSED") or ( not testout.good() ) or ( not testin.good() ) )
      {
        std::cout << "ERROR in " << __FILE__ <<  " failed test write to scratch directory " << scratch << " that's bad. Dying now." << std::endl;
        exit(EXIT_FAILURE);
      }

    }
  }


//  ModelSpace modelspace;

  if (custom_valence_space!="") // if a custom space is defined, the input valence_space is just used as a name
  {
    if (valence_space=="") // if no name is given, then just name it "custom"
    {
      parameters.string_par["valence_space"] = "custom";
      flowfile = parameters.DefaultFlowFile();
      intfile = parameters.DefaultIntFile();
    }
    valence_space = custom_valence_space;
  }


  ModelSpace modelspace = ( reference=="default" ? ModelSpace(eMax,valence_space) : ModelSpace(eMax,reference,valence_space) );

//  std::cout << __LINE__ << "  constructed modelspace " << std::endl;
  modelspace.SetE3max(E3max);
  modelspace.SetLmax(lmax);

//  std::cout << __LINE__ << "  done setting E3max and lmax " << std::endl;


  if (emax_unocc>0)
  {
    modelspace.SetEmaxUnocc(emax_unocc);
  }

  if (physical_system == "atomic")
  {
    modelspace.InitSingleSpecies(eMax, reference, valence_space);
  }

  if (occ_file != "none" and occ_file != "" )
  {
    modelspace.Init_occ_from_file(eMax,valence_space,occ_file);
  }


  if (nsteps < 0) // default to 1 step for single ref, 2 steps for valence decoupling
    nsteps = modelspace.valence.size()>0 ? 2 : 1;


  modelspace.SetHbarOmega(hw);
  if (targetMass>0)
     modelspace.SetTargetMass(targetMass);
  if (lmax3>0)
     modelspace.SetLmax3(lmax3);



// For both dagger operators and single particle wave functions, it's convenient to
// just get every orbit in the valence space. So if SPWF="valence" ,  we append all valence orbits
  if ( std::find( spwf.begin(), spwf.end(), "valence" ) != spwf.end() )
  {
    // this erase/remove idiom is needed because remove just shuffles things around rather than actually removing it.
    spwf.erase( std::remove( spwf.begin(), spwf.end(), "valence" ), std::end(spwf) );
    for ( auto v : modelspace.valence )
    {
      spwf.push_back( modelspace.Index2String(v) );
    }
  }

  if ( std::find( opnames.begin(), opnames.end(), "rhop_all") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "rhop_all"), std::end(opnames) );
    for ( double r=0.0; r<=10.0; r+=0.2 )
    {
       std::ostringstream opn;
       opn << "rhop_" << r;
       opnames.push_back( opn.str() );
    }
  }

  if ( std::find( opnames.begin(), opnames.end(), "rhon_all") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "rhon_all"), std::end(opnames) );
    for ( double r=0.0; r<=10.0; r+=0.2 )
    {
       std::ostringstream opn;
       opn << "rhon_" << r;
       opnames.push_back( opn.str() );
    }
  }

  if ( std::find( opnames.begin(), opnames.end(), "DaggerHF_valence") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "DaggerHF_valence"), std::end(opnames) );
    for ( auto v : modelspace.valence )
    {
      opnames.push_back( "DaggerHF_"+modelspace.Index2String(v) );
    }
    std::cout << "I found DaggerHF_valence, so I'm changing the opnames list to :" << std::endl;
    for ( auto opn : opnames ) std::cout << opn << " ,  ";
    std::cout << std::endl;
  }

  if ( std::find( opnames.begin(), opnames.end(), "DaggerAlln_valence") != opnames.end() )
  {
    opnames.erase( std::remove( opnames.begin(), opnames.end(), "DaggerAlln_valence"), std::end(opnames) );
    for ( auto v : modelspace.valence )
    {
      opnames.push_back( "DaggerAlln_"+modelspace.Index2String(v) );
    }
    std::cout << "I found DaggerAlln_valence, so I'm changing the opnames list to :" << std::endl;
    for ( auto opn : opnames ) std::cout << opn << " ,  ";
    std::cout << std::endl;
  }


//  std::cout << "Making the Hamiltonian..." << std::endl;
  int particle_rank = input3bme=="none" ? 2 : 3;
  Operator Hbare = Operator(modelspace,0,0,0,particle_rank);
  Hbare.SetHermitian();
  Operator VNN = Operator(modelspace,0,0,0,particle_rank);
  VNN.SetHermitian();
  Operator V3N_NO2B = Operator(modelspace,0,0,0,particle_rank);
  V3N_NO2B.SetHermitian();
  Operator V3N = Operator(modelspace,0,0,0,particle_rank);
  V3N.SetHermitian();
  Operator Trel = Operator(modelspace,0,0,0,particle_rank);
  Trel.SetHermitian();


  Commutator::SetUseGooseTank(goose_tank);
  Commutator::SetThreebodyThreshold(threebody_threshold);

  std::cout << "Reading interactions..." << std::endl;


  if (inputtbme != "none")
  {
    if (fmt2 == "me2j")
      rw.ReadBareTBME_Darmstadt(inputtbme, VNN,file2e1max,file2e2max,file2lmax);
    else if (fmt2 == "navratil" or fmt2 == "Navratil")
      rw.ReadBareTBME_Navratil(inputtbme, VNN);
    else if (fmt2 == "oslo" )
      rw.ReadTBME_Oslo(inputtbme, VNN);
    else if (fmt2.find("oakridge") != std::string::npos )
    { // input format should be: singleparticle.dat,vnn.dat
      size_t comma_pos = inputtbme.find_first_of(",");
      if ( fmt2.find("bin") != std::string::npos )
        rw.ReadTBME_OakRidge( inputtbme.substr(0,comma_pos),  inputtbme.substr( comma_pos+1 ), VNN, "binary");
      else
        rw.ReadTBME_OakRidge( inputtbme.substr(0,comma_pos),  inputtbme.substr( comma_pos+1 ), VNN, "ascii");
    }
    else if (fmt2 == "takayuki" )
      rw.ReadTwoBody_Takayuki( inputtbme, VNN);
    else if (fmt2 == "nushellx" )
      rw.ReadNuShellX_int( VNN, inputtbme );
    else if (fmt2 == "schematic" )
    {
      std::cout << "using schematic potential " << inputtbme << std::endl;
      if ( inputtbme == "Minnesota") VNN += imsrg_util::MinnesotaPotential( modelspace );
    }

    Hbare += VNN;
    std::cout << "done reading 2N" << std::endl;
  }

  // Read in the 3-body file
  if (Hbare.particle_rank >=3)
  {
    if(input3bme_type == "full")
    {
      rw.Read_Darmstadt_3body(input3bme, V3N, file3e1max,file3e2max,file3e3max);
    }
    if(input3bme_type == "no2b")
    {

      V3N.ThreeBody.SetMode("no2b");
      if (no2b_precision == "half")  V3N.ThreeBody.SetMode("no2bhalf");
      Hbare.ThreeBody.SetMode("no2b");
      if (no2b_precision == "half")  Hbare.ThreeBody.SetMode("no2bhalf");

      V3N.ThreeBody.ReadFile( {input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max} );
      rw.File3N = input3bme;

    }
    else if(input3bme_type == "mono")
    {
      V3N.ThreeBody.SetMode("mono");
      Hbare.ThreeBody.SetMode("mono");
      V3N.ThreeBody.ReadFile( {input3bme}, {file3e1max, file3e2max, file3e3max, file3e1max} );
      rw.File3N = input3bme;
    }

    Hbare += V3N;
    std::cout << "done reading 3N" << std::endl;
  }

  if (store_3bme_pn)
  {
    Hbare.ThreeBody.TransformToPN();
  }




  if (inputtbme == "none" and physical_system == "atomic")
  {

    using PhysConst::M_ELECTRON;
    using PhysConst::M_NUCLEON;
    int Z = (atomicZ>=0) ?  atomicZ : modelspace.GetTargetZ() ;
    Hbare -= Z*imsrg_util::VCentralCoulomb_Op(modelspace, lmax) * sqrt((M_ELECTRON*1e6)/M_NUCLEON ) ;
    Hbare += imsrg_util::VCoulomb_Op(modelspace, lmax) * sqrt((M_ELECTRON*1e6)/M_NUCLEON ) ;  // convert oscillator length from fm with nucleon mass to nm with electon mass (in eV).
    Hbare += imsrg_util::KineticEnergy_Op(modelspace); // Don't need to rescale this, because it's related to the oscillator frequency, which we input.
    Hbare /= PhysConst::HARTREE; // Convert to Hartree
  }

  if (fmt2 != "nushellx" and physical_system != "atomic" and hw_trap < 0)  // Don't need to add kinetic energy if we read a shell model interaction
  {
    Trel = imsrg_util::Trel_Op(modelspace);
    Hbare += Trel;
    if (Hbare.OneBody.has_nan())
    {
      std::cout << "  Looks like the Trel op is hosed from the get go. Dying." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Add an external harmonic trap
  if ( hw_trap > 0 )
  {
    Hbare += 0.5 * (PhysConst::M_NUCLEON * hw_trap * hw_trap)/(PhysConst::HBARC*PhysConst::HBARC) * imsrg_util::RSquaredOp(modelspace); 
    Hbare += imsrg_util::KineticEnergy_Op(modelspace); // use lab-frame kinetic energy
  }

  // correction to kinetic energy because M_proton != M_neutron
  if ( nucleon_mass_correction)
  {
    Hbare += imsrg_util::Trel_Masscorrection_Op(modelspace);
  }

  if ( relativistic_correction)
  {
    Hbare += imsrg_util::KineticEnergy_RelativisticCorr(modelspace);
  }




  // Add a Lawson center of mass term. If hwBetaCM is specified, use that frequency, otherwise use the basis frequency
  if (std::abs(BetaCM)>1e-3)
  {
    if (hwBetaCM < 0) hwBetaCM = modelspace.GetHbarOmega();
    std::ostringstream hcm_opname;
    hcm_opname << "HCM_" << hwBetaCM;
    Hbare += BetaCM * imsrg_util::OperatorFromString( modelspace, hcm_opname.str());
  }




  std::cout << "Creating HF" << std::endl;
  HFMBPT hf(Hbare); // HFMBPT inherits from HartreeFock, so this works for HF and NAT bases.

  if (not freeze_occupations )  hf.UnFreezeOccupations();
  if ( discard_no2b_from_3n) hf.DiscardNO2Bfrom3N();
  std::cout << "Solving" << std::endl;

  if (basis!="oscillator")
  {
    hf.Solve();
    std::cout << "HF is solved! Rejoice!" << std::endl;
  }

  // decide what to keep after normal ordering
  int hno_particle_rank = 2;
  if ((IMSRG3) and (Hbare.ThreeBodyNorm() > 1e-5))  hno_particle_rank = 3;
  if (discard_residual_input3N) hno_particle_rank = 2;
  if (input3bme_type=="no2b") hno_particle_rank = 2;

  Operator& HNO = Hbare; // The reference & means we overwrite Hbare and save some memory
  if (basis == "HF" and method !="HF")
  {
    std::cout << "Normal ordering H!" << std::endl;
    HNO = hf.GetNormalOrderedH( hno_particle_rank );
    if ((IMSRG3 or perturbative_triples) and OccNat3Cut>0 ) hf.GetNaturalOrbitals();
    std::cout << "H is normal ordered! Rejoice!" << std::endl;
  }
  else if (basis == "NAT") // we want to use the natural orbital basis
  {
    // for backwards compatibility: order_NAT_by_energy overrides NAT_order
    if (order_NAT_by_energy) NAT_order = "energy";

    hf.UseNATOccupations( use_NAT_occupations );
    hf.OrderNATBy( NAT_order );

  //  GetNaturalOrbitals() calls GetDensityMatrix(), which computes the 1b density matrix up to MBPT2
  //  using the NO2B Hamiltonian in the HF basis, obtained with GetNormalOrderedH().
  //  Then it calls DiagonalizeRho() which diagonalizes the density matrix, yielding the natural orbital basis.
    hf.GetNaturalOrbitals();
    if (use_HF_reference_in_NAT) {
      if (use_HF_valence_in_NAT) {
        hf.UseHFForHoleAndValenceStates();
      } else {
        hf.UseHFForHoleStates();
      }
    }
    HNO = hf.GetNormalOrderedHNAT( hno_particle_rank );

  //  SRS: I'm commenting this out because this is not reasonably-expected default behavior
  //    // For now, even if we use the NAT occupations, we switch back to naive occupations after the normal ordering
  //    // This should be investigated in more detail.
  //    if (use_NAT_occupations)
  //    {
  //      hf.FillLowestOrbits();
  //      std::cout << "Undoing NO wrt A=" << modelspace.GetAref() << " Z=" << modelspace.GetZref() << std::endl;
  //      HNO = HNO.UndoNormalOrdering();
  //      hf.UpdateReference();
  //      modelspace.SetReference(modelspace.core); // change the reference
  //      std::cout << "Doing NO wrt A=" << modelspace.GetAref() << " Z=" << modelspace.GetZref() << std::endl;
  //      HNO = HNO.DoNormalOrdering();
  //    }

  }
  else if (basis == "oscillator")
  {
    HNO = Hbare.DoNormalOrdering();
  }

  std::cout << "Clearing V3N!" << std::endl;
  V3N *= 0.0;
  if (input3bme != "none") {
    std::cout << "Handling input 3BME!" << std::endl;
    VNN.SetNumberLegs(4);
    VNN.SetParticleRank(2);
    Trel.SetNumberLegs(4);
    Trel.SetParticleRank(2);
    std::cout << "Transforming Trel and VNN to HF!" << std::endl;
    Operator VNN_Trans = hf.TransformToHFBasis(VNN);
    Operator Trel_Trans = hf.TransformToHFBasis(Trel);
    std::cout << "Extracting V3N from HNO in HF!" << std::endl;
    Operator V3N_TransNO = HNO;
    V3N_TransNO.SetNumberLegs(4);
    V3N_TransNO.SetParticleRank(2);

    std::cout << "Removing VNN and Trel in HF!" << std::endl;
    V3N_TransNO -= VNN_Trans;
    V3N_TransNO -= Trel_Trans;

    if (inputtbmeNO2B != "none") {
      std::cout << "Reading Jacobi NO NO2B 3N files!" << std::endl;
      rw.ReadBareTBME_np_Darmstadt(inputtbmeNO2B, V3N_NO2B, eMax, 2 * eMax, eMax);
      V3N_NO2B.SetNumberLegs(4);
      V3N_NO2B.SetParticleRank(2);
      std::cout << "Clearing and replacing V3N NO2B part!" << std::endl;
      V3N_TransNO.TwoBody *= 0.0;
      V3N_TransNO += hf.TransformToHFBasis(V3N_NO2B);
    }

    std::cout << "Undoing normal ordering in HF!" << std::endl;
    Operator V3N_Trans = V3N_TransNO.UndoNormalOrdering();

    std::cout << "Transforming V3N from HF to HO!" << std::endl;
    V3N = hf.TransformFromHFBasis(V3N_Trans);

    std::string name_prefix = parameters.s("name_prefix");
    if (name_prefix == "default") {
      name_prefix = "NO2B_3BME_" + reference + "_hw_" + std::to_string(hw) + "_e_" + std::to_string(eMax) + "_E3_" + std::to_string(E3max);
      if (inputtbmeNO2B != "none") {
        name_prefix = "JacobiNO_" + name_prefix;
      }
    }

    std::cout << "Writing to file " << name_prefix << "_xb.ornlme !" <<  std::endl;
    if (parameters.s("2bme_output_type") == "binary") {
      rw.WriteOakRidgeFull(parameters.s("spb_file"), name_prefix + "_0b.ornlme", name_prefix + "_1b.ornlme", name_prefix + "_2b.ornlme.bin", V3N, "binary");
    } else {
      rw.WriteOakRidgeFull(parameters.s("spb_file"), name_prefix + "_0b.ornlme", name_prefix + "_1b.ornlme", name_prefix + "_2b.ornlme", V3N, "not_binary");
    }

  }

  Hbare.PrintTimes();

  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

