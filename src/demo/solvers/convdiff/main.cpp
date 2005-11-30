// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002-11-29
// Last changed: 2005-11-29
//
// A simple test program for convection-diffusion, solving
//
//     du/dt + b.grad u - div a grad u = f
//
// around a hot dolphin in 2D, with diffusivity given by
//
//     a(x,y,t) = 0.1
//
// and convection given by
//
//     b(x,y,t) = (-5,0).
//
// This program illustrates the need for stabilisation, for
// instance streamline-diffusion, for large values of b. For
// |b| > 10 oscillations start to appear. Try b = (-100,0)
// to see some quite large oscillations.

#include <dolfin.h>

using namespace dolfin;

// Convection
class Convection : public Function
{
  real operator() (const Point& p, unsigned int i)
  {
    if ( i == 0 )
      return -5.0;
    else
      return 0.0;
  }
};

// Right-hand side
class Source : public Function
{
  real operator() (const Point& p, unsigned int i)
  {
    return 0.0;
  }
};

// Boundary condition
class MyBC : public BoundaryCondition
{
  const BoundaryValue operator() (const Point& p)
  {
    BoundaryValue value;

    if ( p.x == 1.0 )
      value.set(0.0);
    else if ( p.x != 0.0 && p.x != 1.0 && p.y != 0.0 && p.y != 1.0 )
      value.set(1.0);
    
    return value;
  }
};

int main()
{
  dolfin_output("curses");

  Mesh mesh("dolfin.xml.gz");
  Convection w;
  Source f;
  MyBC bc;

  ConvectionDiffusionSolver::solve(mesh, w, f, bc);
  
  return 0;
}
