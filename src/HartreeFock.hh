#ifndef HartreeFock_h
#define HartreeFock_h

#include "Operator.hh"
#include <armadillo>
#include <map>

class HartreeFock
{
 public:
   Operator& Hbare;         ///< Input bare Hamiltonian
   arma::mat C;             ///< coefficients, 1st index is ho basis, 2nd = HF basis
   arma::mat rho;           ///< density matrix rho_ij
   arma::mat t;             ///< kinetic energy
   arma::mat Vij;           ///< 1 body piece of 2 body potential
   arma::mat V3ij;          ///< 1 body piece of 3 body potential
   arma::mat F;             ///< Fock matrix
   arma::mat Vmon;          ///< Monopole 2-body interaction
   arma::mat Vmon_exch;     ///< Monopole 2-body interaction
   arma::vec energies;      ///< vector of single particle energies
   arma::vec prev_energies; ///< SPE's from last iteration
   double tolerance;        ///< tolerance for convergence
   double EHF;              ///< Hartree-Fock energy (Normal-ordered 0-body term)
   int iterations;          ///< iterations used in Solve()

   map<unsigned long long int, double> Vmon3; // Monopole 3-body interaction
   

   HartreeFock(Operator&  hbare);
   void BuildMonopoleV();
   void Diagonalize();
   void UpdateF();
   void UpdateDensityMatrix();
   bool CheckConvergence();
   void Solve();
   void CalcEHF();
   void ReorderCoefficients();
   Operator TransformToHFBasis( Operator& OpIn);
   Operator GetNormalOrderedH();

   void BuildMonopoleV3();

   Operator GetHbare(){return Hbare;};

};



#endif
