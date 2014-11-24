
#include "IMSRGSolver.hh"



// Constructor
IMSRGSolver::IMSRGSolver(Operator H_in)
{
   method = "BCH";
   generator = "white";
   s = 0;
   ds = 0.1;
   ds_max = 1.0;
   smax  = 2.0;
   i_full_BCH = 5;
   norm_domega = 0.1;
   H_0 = H_in;
   H_s = H_in;
   Eta = H_in;
   Eta.EraseZeroBody();
   Eta.EraseOneBody();
   Eta.EraseTwoBody();
   Eta.SetAntiHermitian();
   modelspace = H_0.GetModelSpace();
   Omega = H_s;
   Omega.EraseZeroBody();
   Omega.EraseOneBody();
   Omega.EraseTwoBody();
   Omega.SetAntiHermitian();
   dOmega = Omega;
}



void IMSRGSolver::Solve()
{
   // If we have a flow output file, open it up and write to it here.
   ofstream flowf;
   if (flowfile != "")
      flowf.open(flowfile,ofstream::out);
   cout << " i     s       E0       ||H_1||      ||H_2||        ||Omega||     || Eta_1||    || Eta_2 ||    ||dOmega||     " << endl;


   double norm_eta,norm_eta_last;

   for (istep=0;s<smax;++istep)
   {

      UpdateEta();

      // Write details of the flow
      WriteFlowStatus(flowf);
      WriteFlowStatus(cout);

      norm_eta = Eta.Norm();
      // ds should never be more than 1, as this is over-rotating
//      ds = min(norm_domega / norm_eta, ds_max); 
//      if (ds == ds_max) norm_domega /=2;
//      if (s+ds > smax) ds = smax-s;
      s += ds;
      dOmega = Eta * ds; // Here's the Euler step.

      // accumulated generator (aka Magnus operator) exp(Omega) = exp(dOmega) * exp(Omega_last)
      Omega = dOmega.BCH_Product( Omega ); 

      // transformed Hamiltonian H_s = exp(Omega) H_0 exp(-Omega)
      if (istep%i_full_BCH == i_full_BCH-1)
      {
         H_s = H_0.BCH_Transform( Omega );   
      }
      else
      {
         H_s = H_s.BCH_Transform( dOmega );  // less accurate, but converges with fewer commutators, since ||dOmega|| < ||Omega||
      }




   }
   // if the last calculation of H_s was the quick way,
   // do it again the more accurate way.
   if (istep%i_full_BCH != i_full_BCH-1)
   {
      H_s = H_0.BCH_Transform( Omega ); 
      WriteFlowStatus(flowf);
      WriteFlowStatus(cout);
   }

   if (flowfile != "")
      flowf.close();

}



// Returns exp(Omega) OpIn exp(-Omega)
Operator IMSRGSolver::Transform(Operator& OpIn)
{
   return OpIn.BCH_Transform( Omega );
}


void IMSRGSolver::UpdateEta()
{
   
   if (generator == "wegner")
   {
     ConstructGenerator_Wegner();
   } 
   else if (generator == "white")
   {
     ConstructGenerator_White();
   } 
   else if (generator == "atan")
   {
     ConstructGenerator_Atan();
   } 
   else if (generator == "shell-model")
   {
     ConstructGenerator_ShellModel();
   }
   else if (generator == "shell-model-atan")
   {
     ConstructGenerator_ShellModel_Atan();
   }
   else if (generator == "shell-model-1hw") // Doesn't work yet
   {
     ConstructGenerator_ShellModel1hw(); // Doesn't work yet
   }
   else
   {
      cout << "Error. Unkown generator: " << generator << endl;
   }

}



// Epstein-Nesbet energy denominators for White-type generators
double IMSRGSolver::GetEpsteinNesbet1bDenominator(int i, int j)
{
   double denominator = H_s.OneBody(i,i) - H_s.OneBody(j,j) - H_s.GetTBMEmonopole(j,i,j,i);
   return denominator;
}



// This could likely be sped up by constructing and storing the monopole matrix
double IMSRGSolver::GetEpsteinNesbet2bDenominator(int ch, int ibra, int iket)
{
   TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
   Ket * bra = tbc.GetKet(ibra);
   Ket * ket = tbc.GetKet(iket);
   int i = bra->p;
   int j = bra->q;
   int a = ket->p;
   int b = ket->q;
   double denominator = H_s.GetTBMEmonopole(i,j,i,j); // pp'pp'
   denominator       += H_s.GetTBMEmonopole(a,b,a,b); // hh'hh'
   denominator       -= H_s.GetTBMEmonopole(i,a,i,a); // phph
   denominator       -= H_s.GetTBMEmonopole(i,b,i,b); // ph'ph'
   denominator       -= H_s.GetTBMEmonopole(j,a,j,a); // p'hp'h
   denominator       -= H_s.GetTBMEmonopole(j,b,j,b); // p'h'p'h'

   denominator += H_s.OneBody(i,i)+ H_s.OneBody(j,j) - H_s.OneBody(a,a) - H_s.OneBody(b,b);

//   if (abs(denominator ) < 0.01)
//      cout << "2b denominator "  << ch << " " << ibra << "," << iket << " = " << denominator << endl;
   return denominator;
}



void IMSRGSolver::ConstructGenerator_Wegner()
{
   H_diag = H_s;
   H_diag.ZeroBody = 0;
   for (int &a : modelspace->holes)
   {
      for (int &b : modelspace->valence)
      {
         H_diag.OneBody(a,b) =0;
         H_diag.OneBody(b,a) =0;
      }
   }

   for (int ch=0;ch<modelspace->GetNumberTwoBodyChannels();++ch)
   {  // Note, should also decouple the v and q spaces
      // This is wrong. The projection operator should be different.
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      H_diag.TwoBody[ch] = (tbc.Proj_hh*H_diag.TwoBody[ch] + tbc.Proj_pp*H_diag.TwoBody[ch]);
   }

   Eta = H_diag.Commutator(H_s);
}



void IMSRGSolver::ConstructGenerator_White()
{
   // One body piece -- eliminate ph bits
   for ( int &i : modelspace->particles)
   {
      Orbit *oi = modelspace->GetOrbit(i);
      for (int &a : modelspace->holes)
      {
         Orbit *oa = modelspace->GetOrbit(a);
         double denominator = GetEpsteinNesbet1bDenominator(i,a);
         Eta.OneBody(i,a) = H_s.OneBody(i,a)/denominator;
         Eta.OneBody(a,i) = - Eta.OneBody(i,a);
      }
   }

   // Two body piece -- eliminate pp'hh' bits. Note that the hh'hp pieces are accounted
   // for in the normal ordered one-body part.
   int nchan = modelspace->GetNumberTwoBodyChannels();
   for (int ch=0;ch<nchan;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      for (int& ibra : tbc.KetIndex_hh)
      {
         for (int& iket : tbc.KetIndex_pp)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }
    }
}







void IMSRGSolver::ConstructGenerator_ShellModel()
{
//   ConstructGenerator_White(); // Start with the White generator

   // One body piece -- make sure the valence one-body part is diagonal
   for ( int &i : modelspace->valence)
   {
      Orbit *oi = modelspace->GetOrbit(i);
      for (int j=0; j<modelspace->GetNumberOrbits(); ++j)
//      for (int &j : modelspace->particles)
      {
         if (i==j) continue;
         Orbit *oj = modelspace->GetOrbit(j);
         double denominator = GetEpsteinNesbet1bDenominator(i,j);
         Eta.OneBody(i,j) = H_s.OneBody(i,j)/denominator;
         Eta.OneBody(j,i) = - Eta.OneBody(i,j);
      }
   
   }
   // Two body piece -- eliminate ppvh and pqvv  ( vv'hh' was already accounted for with White )
   // This is still no good...

   int nchan = modelspace->GetNumberTwoBodyChannels();
   for (int ch=0;ch<nchan;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);

      // Decouple vv from qq and qv

      for (int& ibra : tbc.KetIndex_vv)
      {
//         for (int& iket : tbc.KetIndex_qq) // this includes vv qh  which we don't want to include...
         for (int& iket : tbc.KetIndex_particleq_particleq) 
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }

         for (int& iket : tbc.KetIndex_holeq_holeq) 
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }

         for (int& iket : tbc.KetIndex_v_particleq) 
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }

         for (int& iket : tbc.KetIndex_v_holeq) 
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }


      }


      // Decouple hh states

      for (int& ibra : tbc.KetIndex_holeq_holeq)
      {
         for (int& iket : tbc.KetIndex_particleq_particleq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }

         for (int& iket : tbc.KetIndex_v_particleq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }

      // Decouple vh states

      for (int& ibra : tbc.KetIndex_v_holeq)
      {
         for (int& iket : tbc.KetIndex_particleq_particleq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }

         for (int& iket : tbc.KetIndex_v_particleq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }



    }
}



// This is half-baked and doesn't work yet.
void IMSRGSolver::ConstructGenerator_ShellModel1hw()
{
/*
   ConstructGenerator_White(); // Start with the White generator

   // One body piece -- make sure the valence one-body part is diagonal
   for ( int &i : modelspace->valence)
   {
      Orbit *oi = modelspace->GetOrbit(i);
      for (int &j : modelspace->particles)
      {
         if (i==j) continue;
         Orbit *oj = modelspace->GetOrbit(j);
         double denominator = GetEpsteinNesbet1bDenominator(i,j);
         Eta.OneBody(i,j) = H_s.OneBody(i,j)/denominator;
         Eta.OneBody(j,i) = - Eta.OneBody(i,j);
      }
   
   }
   // Two body piece -- eliminate ppvh and qqvv  ( vv'hh' was already accounted for with White )

   int nchan = modelspace->GetNumberTwoBodyChannels();
   for (int ch=0;ch<nchan;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);

      for (int& ibra : tbc.KetIndex_pp)
      {
         for (int& iket : tbc.KetIndex_vh)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }

      for (int& ibra : tbc.KetIndex_vv)
      {
         for (int& iket : tbc.KetIndex_qq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }

      // Drive diagonal qqqq pieces to high energy to raise the energy of Nhw excitations with N>1
      for (int& ibra : tbc.KetIndex_qq)
      {
        for (int& iket : tbc.KetIndex_qq)
        {
         if (iket == ibra) continue;
         double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
//         Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
         Eta.TwoBody[ch](ibra,iket) = (10.0 - H_s.TwoBody[ch](ibra,iket) )/10.0;//  / denominator;
         //Eta.TwoBody[ch](ibra,iket) = (H_s.TwoBody[ch](ibra,iket) -1.0 )/1.0 ;// / denominator;
         Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
       }  
      }

    }
*/
}



void IMSRGSolver::ConstructGenerator_Atan()
{
   // One body piece -- eliminate ph bits
   for ( int &i : modelspace->particles)
   {
      Orbit *oi = modelspace->GetOrbit(i);
      for (int &a : modelspace->holes)
      {
         Orbit *oa = modelspace->GetOrbit(a);
         double denominator = GetEpsteinNesbet1bDenominator(i,a);
         //Eta.OneBody(i,a) = H_s.OneBody(i,a)/denominator;
         Eta.OneBody(i,a) = 0.5*atan(2*H_s.OneBody(i,a)/denominator);
         Eta.OneBody(a,i) = - Eta.OneBody(i,a);
      }
   }

   // Two body piece -- eliminate pp'hh' bits
   // This could likely be sped up by constructing and storing the monopole matrix
   int nchan = modelspace->GetNumberTwoBodyChannels();
   for (int ch=0;ch<nchan;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);
      for (int& ibra : tbc.KetIndex_pp)
      {
         for (int& iket : tbc.KetIndex_hh)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);

            //Eta.TwoBody[ch](ibra,iket) = H_s.TwoBody[ch](ibra,iket) / denominator;
            Eta.TwoBody[ch](ibra,iket) = 0.5*atan(2*H_s.TwoBody[ch](ibra,iket) / denominator);
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }
    }
}



void IMSRGSolver::ConstructGenerator_ShellModel_Atan()
{
/*
   // One body piece -- make sure the valence one-body part is diagonal
   for ( int &i : modelspace->valence)
   {
      Orbit *oi = modelspace->GetOrbit(i);
      for (int j=0; j<modelspace->GetNumberOrbits(); ++j)
//      for (int &j : modelspace->particles)
      {
         if (i==j) continue;
         Orbit *oj = modelspace->GetOrbit(j);
         double denominator = GetEpsteinNesbet1bDenominator(i,j);
         Eta.OneBody(i,j) = 0.5*atan(2*H_s.OneBody(i,j)/denominator);
         Eta.OneBody(j,i) = - Eta.OneBody(i,j);
      }
   
   }
   // Two body piece -- eliminate ppvh and pqvv  ( vv'hh' was already accounted for with White )

   int nchan = modelspace->GetNumberTwoBodyChannels();
   for (int ch=0;ch<nchan;++ch)
   {
      TwoBodyChannel& tbc = modelspace->GetTwoBodyChannel(ch);


      // Decouple vv from qq and qv

      for (int& ibra : tbc.KetIndex_vv)
      {
         for (int& iket : tbc.KetIndex_qq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = 0.5*atan(2*H_s.TwoBody[ch](ibra,iket) / denominator);
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }


      for (int& ibra : tbc.KetIndex_vv)
      {
         for (int& iket : tbc.KetIndex_vq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = 0.5*atan(2*H_s.TwoBody[ch](ibra,iket) / denominator);
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }


      // Decouple the hole qstates (core) from particle qstates

      for (int& ibra : tbc.KetIndex_holeq_holeq)
      {
         for (int& iket : tbc.KetIndex_particleq_holeq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = 0.5*atan(2*H_s.TwoBody[ch](ibra,iket) / denominator);
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }


      for (int& ibra : tbc.KetIndex_holeq_holeq)
      {
         for (int& iket : tbc.KetIndex_particleq_particleq)
         {
            double denominator = GetEpsteinNesbet2bDenominator(ch,ibra,iket);
            Eta.TwoBody[ch](ibra,iket) = 0.5*atan(2*H_s.TwoBody[ch](ibra,iket) / denominator);
            Eta.TwoBody[ch](iket,ibra) = - Eta.TwoBody[ch](ibra,iket) ; // Eta needs to be antisymmetric
         }
      }

   }
*/
}


void IMSRGSolver::WriteFlowStatus(ostream& f)
{
   if ( f.good() )
   {
      f.width(11);
      f.precision(10);
      f << istep << "      " << s << "      " << H_s.ZeroBody << "     " << H_s.OneBodyNorm() << "    " << H_s.TwoBodyNorm() << "     " << Omega.Norm() << "     " << Eta.OneBodyNorm() << "    " << Eta.TwoBodyNorm() << "   " << dOmega.Norm() << endl;
   }

}

